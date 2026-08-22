//! Deterministic scalar sparse-MoE routing and F32 correctness mechanics.
//!
//! This is a synthetic correctness substrate, not an accepted model
//! architecture or package format. Production activation requires a reviewed
//! architecture ID, exact router semantics, signed tensor inventory, and
//! independent model evidence.

use personalai_engine_model_format::Quantization;
use personalai_engine_placement::{
    PlacementCancellationProbe, SparseExpertExecutionPlan, SparseExpertLease,
    SparseExpertPlanIdentity,
};
use personalai_engine_types::{DType, EngineError, ModelPackageHash};

/// Largest scalar diagnostic dimension. Production execution needs a qualified backend.
pub const MAX_SCALAR_SPARSE_MOE_DIMENSION: usize = 65_536;
/// Scalar correctness work ceiling for one routed layer.
pub const MAX_SCALAR_SPARSE_MOE_OPERATIONS: u64 = 100_000_000;
const MAX_ROUTED_EXPERTS: usize = 4_096;
const MAX_ACTIVE_EXPERTS: usize = 64;

/// One deterministically selected expert and its normalized router weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutedExpert {
    expert: u32,
    weight: f32,
}

impl RoutedExpert {
    /// Zero-based expert ordinal.
    pub const fn expert(self) -> u32 {
        self.expert
    }

    /// Stable-softmax weight over the selected experts.
    pub const fn weight(self) -> f32 {
        self.weight
    }
}

/// Selects exact top-k finite router logits and normalizes only that set.
///
/// Ties select the lower expert ID. The returned vector is ordered by expert
/// ID so expert accumulation order is independent of score-sort behavior.
pub fn route_sparse_experts(
    scores: &[f32],
    active_experts: u16,
) -> Result<Vec<RoutedExpert>, EngineError> {
    if scores.is_empty() || scores.len() > MAX_ROUTED_EXPERTS {
        return Err(EngineError::invalid_input(
            "sparse router expert count is outside hard bounds",
        ));
    }
    let active = usize::from(active_experts);
    if active == 0 || active > MAX_ACTIVE_EXPERTS || active > scores.len() {
        return Err(EngineError::invalid_input("sparse router active-expert count is invalid"));
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(EngineError::backend_failure("sparse router produced a non-finite score"));
    }

    let mut indices = Vec::new();
    indices
        .try_reserve_exact(scores.len())
        .map_err(|_| EngineError::resource_exhaustion("sparse router index allocation failed"))?;
    for index in 0..scores.len() {
        indices.push(index);
    }
    indices.sort_unstable_by(|left, right| {
        let left_bits = scores[*left].to_bits();
        let right_bits = scores[*right].to_bits();
        let left_key = if left_bits.trailing_zeros() >= 31 { 0 } else { left_bits };
        let right_key = if right_bits.trailing_zeros() >= 31 { 0 } else { right_bits };
        if left_key == right_key {
            left.cmp(right)
        } else {
            scores[*right].total_cmp(&scores[*left])
        }
    });
    indices.truncate(active);

    let maximum = indices
        .iter()
        .map(|index| scores[*index])
        .max_by(f32::total_cmp)
        .ok_or_else(|| EngineError::internal_invariant("selected router set is empty"))?;
    let mut exponentials = Vec::new();
    exponentials
        .try_reserve_exact(active)
        .map_err(|_| EngineError::resource_exhaustion("router weight allocation failed"))?;
    let mut total = 0.0_f64;
    for index in &indices {
        let value = (f64::from(scores[*index]) - f64::from(maximum)).exp();
        if !value.is_finite() || value < 0.0 {
            return Err(EngineError::backend_failure("sparse router normalization is invalid"));
        }
        total += value;
        exponentials.push(value);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(EngineError::backend_failure("sparse router normalization is invalid"));
    }

    let mut routed = Vec::new();
    routed
        .try_reserve_exact(active)
        .map_err(|_| EngineError::resource_exhaustion("routed expert allocation failed"))?;
    for (index, exponential) in indices.into_iter().zip(exponentials) {
        let weight = f32_from_f64(exponential / total, "sparse router weight exceeds F32")?;
        if !weight.is_finite() || weight < 0.0 {
            return Err(EngineError::backend_failure("sparse router weight is invalid"));
        }
        routed.push(RoutedExpert {
            expert: u32::try_from(index)
                .map_err(|_| EngineError::resource_exhaustion("expert index overflows"))?,
            weight,
        });
    }
    routed.sort_unstable_by_key(|expert| expert.expert);
    Ok(routed)
}

/// Resident router plus scalar F32 expert execution semantics for tiny fixtures.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarSparseMoeLayer {
    hidden_dimension: usize,
    intermediate_dimension: usize,
    expert_count: u32,
    active_experts: u16,
    router: Box<[f32]>,
    expert_payload_bytes: usize,
    binding: Option<ScalarSparseMoeBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScalarSparseMoeBinding {
    identity: SparseExpertPlanIdentity,
    plan_digest: ModelPackageHash,
    layer: u32,
}

impl ScalarSparseMoeLayer {
    /// Constructs bounded row-major router weights `[expert, hidden]`.
    pub fn new(
        hidden_dimension: usize,
        intermediate_dimension: usize,
        expert_count: u32,
        active_experts: u16,
        router: Vec<f32>,
    ) -> Result<Self, EngineError> {
        if hidden_dimension == 0
            || intermediate_dimension == 0
            || hidden_dimension > MAX_SCALAR_SPARSE_MOE_DIMENSION
            || intermediate_dimension > MAX_SCALAR_SPARSE_MOE_DIMENSION
        {
            return Err(EngineError::invalid_input(
                "scalar sparse-MoE dimension is outside hard bounds",
            ));
        }
        let expert_count_usize = usize::try_from(expert_count)
            .map_err(|_| EngineError::resource_exhaustion("expert count overflows"))?;
        if expert_count_usize == 0 || expert_count_usize > MAX_ROUTED_EXPERTS {
            return Err(EngineError::invalid_input(
                "scalar sparse-MoE expert count is outside hard bounds",
            ));
        }
        if active_experts == 0
            || usize::from(active_experts) > MAX_ACTIVE_EXPERTS
            || usize::from(active_experts) > expert_count_usize
        {
            return Err(EngineError::invalid_input(
                "scalar sparse-MoE active-expert count is invalid",
            ));
        }
        let router_elements = expert_count_usize
            .checked_mul(hidden_dimension)
            .ok_or_else(|| EngineError::resource_exhaustion("router shape overflows"))?;
        if router.len() != router_elements || router.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_input("router weights are invalid"));
        }
        let matrix_elements = intermediate_dimension
            .checked_mul(hidden_dimension)
            .ok_or_else(|| EngineError::resource_exhaustion("expert matrix shape overflows"))?;
        let down_elements = hidden_dimension
            .checked_mul(intermediate_dimension)
            .ok_or_else(|| EngineError::resource_exhaustion("expert down shape overflows"))?;
        let expert_elements = matrix_elements
            .checked_mul(2)
            .and_then(|value| value.checked_add(down_elements))
            .ok_or_else(|| EngineError::resource_exhaustion("expert payload shape overflows"))?;
        let expert_payload_bytes = expert_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| EngineError::resource_exhaustion("expert payload bytes overflow"))?;
        let operations = u64::try_from(router_elements)
            .ok()
            .and_then(|router_ops| {
                u64::try_from(expert_elements)
                    .ok()
                    .and_then(|expert_ops| expert_ops.checked_mul(u64::from(active_experts)))
                    .and_then(|expert_ops| router_ops.checked_add(expert_ops))
            })
            .ok_or_else(|| {
                EngineError::resource_exhaustion("sparse-MoE operation count overflows")
            })?;
        if operations > MAX_SCALAR_SPARSE_MOE_OPERATIONS {
            return Err(EngineError::resource_exhaustion(
                "scalar sparse-MoE operation ceiling exceeded",
            ));
        }
        Ok(Self {
            hidden_dimension,
            intermediate_dimension,
            expert_count,
            active_experts,
            router: router.into_boxed_slice(),
            expert_payload_bytes,
            binding: None,
        })
    }

    /// Binds this diagnostic evaluator to one exact immutable expert plan/layer.
    pub fn bind_to_plan(
        mut self,
        plan: &SparseExpertExecutionPlan,
        layer: u32,
    ) -> Result<Self, EngineError> {
        if self.binding.is_some() {
            return Err(EngineError::internal_invariant(
                "scalar sparse-MoE evaluator is already bound",
            ));
        }
        let layer_plan = plan
            .layers()
            .iter()
            .find(|candidate| candidate.layer() == layer)
            .ok_or_else(|| EngineError::invalid_input("sparse expert layer is absent"))?;
        if layer_plan.experts().len()
            != usize::try_from(self.expert_count)
                .map_err(|_| EngineError::resource_exhaustion("expert count overflows"))?
            || layer_plan.active_experts() != self.active_experts
        {
            return Err(EngineError::invalid_input(
                "scalar sparse-MoE evaluator does not match the immutable layer plan",
            ));
        }
        self.binding = Some(ScalarSparseMoeBinding {
            identity: plan.identity().clone(),
            plan_digest: plan.plan_digest(),
            layer,
        });
        Ok(self)
    }

    /// Exact opaque F32 fixture payload length for one gate/up/down expert.
    pub const fn expert_payload_bytes(&self) -> usize {
        self.expert_payload_bytes
    }

    /// Computes deterministic router selection from one finite hidden vector.
    pub fn route(&self, input: &[f32]) -> Result<Vec<RoutedExpert>, EngineError> {
        self.route_with_cancellation(input, &|| false)
    }

    fn route_with_cancellation(
        &self,
        input: &[f32],
        is_cancelled: &impl Fn() -> bool,
    ) -> Result<Vec<RoutedExpert>, EngineError> {
        self.validate_input(input)?;
        let expert_count = usize::try_from(self.expert_count)
            .map_err(|_| EngineError::resource_exhaustion("expert count overflows"))?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(expert_count)
            .map_err(|_| EngineError::resource_exhaustion("router score allocation failed"))?;
        for expert in 0..expert_count {
            check_execution_cancelled(is_cancelled)?;
            let row = expert
                .checked_mul(self.hidden_dimension)
                .ok_or_else(|| EngineError::resource_exhaustion("router row overflows"))?;
            let mut score = 0.0_f64;
            for (column, input_value) in input.iter().enumerate() {
                score += f64::from(self.router[row + column]) * f64::from(*input_value);
            }
            scores.push(f32_from_f64(score, "router score is non-finite")?);
        }
        route_sparse_experts(&scores, self.active_experts)
    }

    /// Executes verified cache leases only when route, layer, and plan identity match.
    pub fn execute_cached_f32(
        &self,
        input: &[f32],
        leases: &[SparseExpertLease],
        cancellation: &dyn PlacementCancellationProbe,
    ) -> Result<Vec<f32>, EngineError> {
        let binding = self.binding.as_ref().ok_or_else(|| {
            EngineError::not_implemented(
                "scalar sparse-MoE cache execution requires an exact plan binding",
            )
        })?;
        let expected_identity = &binding.identity;
        let expected_plan_digest = binding.plan_digest;
        let layer = binding.layer;
        if expected_identity.quantization() != Quantization::None {
            return Err(EngineError::not_implemented(
                "scalar sparse-MoE execution supports only unquantized F32 expert storage",
            ));
        }
        let lifecycle_lease = leases.first();
        let is_cancelled =
            || cancellation.is_cancelled() || leases.iter().any(SparseExpertLease::is_cancelled);
        let routed = self.route_with_cancellation(input, &is_cancelled)?;
        if leases.len() != routed.len() {
            return Err(EngineError::authentication_failure(
                "sparse expert lease count does not match the trusted route",
            ));
        }
        if leases.iter().any(|lease| lease.storage_dtype() != DType::F32) {
            return Err(EngineError::not_implemented(
                "scalar sparse-MoE execution supports only F32 expert storage",
            ));
        }
        if lifecycle_lease.is_some_and(|first| {
            leases.iter().any(|lease| !first.shares_cache_instance_with(lease))
        }) {
            return Err(EngineError::authentication_failure(
                "sparse expert leases belong to different cache instances",
            ));
        }
        for (routed_expert, lease) in routed.iter().zip(leases) {
            if lease.identity() != expected_identity
                || lease.plan_digest() != expected_plan_digest
                || lease.id().layer() != layer
                || lease.id().expert() != routed_expert.expert
            {
                return Err(EngineError::authentication_failure(
                    "sparse expert lease does not match the trusted route",
                ));
            }
        }
        let first_lease = leases.first().ok_or_else(|| {
            EngineError::internal_invariant("validated sparse expert lease set is empty")
        })?;
        let mut scratch = first_lease
            .allocate_execution_staging(self.scratch_bytes()?)
            .map_err(personalai_engine_placement::PlacementError::into_engine_error)?;
        self.execute_routed_f32(input, &routed, scratch.as_mut_slice(), &is_cancelled, |expert| {
            leases
                .iter()
                .find(|lease| lease.id().expert() == expert)
                .map(SparseExpertLease::bytes)
                .ok_or_else(|| {
                    EngineError::internal_invariant("validated sparse expert lease disappeared")
                })
        })
    }

    #[cfg(test)]
    fn execute_f32_payload<'a>(
        &self,
        input: &[f32],
        expert_payload: impl FnMut(u32) -> Result<&'a [u8], EngineError>,
    ) -> Result<Vec<f32>, EngineError> {
        let routed = self.route(input)?;
        let mut scratch = fallible_zeroed_bytes(
            self.scratch_bytes()?,
            "scalar sparse-MoE scratch allocation failed",
        )?;
        self.execute_routed_f32(input, &routed, &mut scratch, &|| false, expert_payload)
    }

    fn execute_routed_f32<'a>(
        &self,
        input: &[f32],
        routed: &[RoutedExpert],
        scratch: &mut [u8],
        is_cancelled: &impl Fn() -> bool,
        mut expert_payload: impl FnMut(u32) -> Result<&'a [u8], EngineError>,
    ) -> Result<Vec<f32>, EngineError> {
        if scratch.len() != self.scratch_bytes()? {
            return Err(EngineError::internal_invariant(
                "scalar sparse-MoE staging length is invalid",
            ));
        }
        scratch.fill(0);
        let matrix_elements = self
            .intermediate_dimension
            .checked_mul(self.hidden_dimension)
            .ok_or_else(|| EngineError::resource_exhaustion("expert matrix shape overflows"))?;
        let up_offset = matrix_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| EngineError::resource_exhaustion("expert up offset overflows"))?;
        let down_offset = up_offset
            .checked_mul(2)
            .ok_or_else(|| EngineError::resource_exhaustion("expert down offset overflows"))?;

        for routed_expert in routed {
            check_execution_cancelled(is_cancelled)?;
            let bytes = expert_payload(routed_expert.expert)?;
            if bytes.len() != self.expert_payload_bytes {
                return Err(EngineError::corrupt_model_package(
                    "scalar sparse-MoE expert payload length is invalid",
                ));
            }
            for intermediate in 0..self.intermediate_dimension {
                check_execution_cancelled(is_cancelled)?;
                let row = intermediate
                    .checked_mul(self.hidden_dimension)
                    .ok_or_else(|| EngineError::resource_exhaustion("expert row overflows"))?;
                let mut gate = 0.0_f64;
                let mut up = 0.0_f64;
                for (column, input_value) in input.iter().enumerate() {
                    let element = row.checked_add(column).ok_or_else(|| {
                        EngineError::resource_exhaustion("expert index overflows")
                    })?;
                    gate += f64::from(read_f32(bytes, 0, element)?) * f64::from(*input_value);
                    up += f64::from(read_f32(bytes, up_offset, element)?) * f64::from(*input_value);
                }
                if !gate.is_finite() || !up.is_finite() {
                    return Err(EngineError::backend_failure(
                        "scalar sparse-MoE projection is non-finite",
                    ));
                }
                let silu = gate / (1.0 + (-gate).exp());
                let value = silu * up;
                if !value.is_finite() {
                    return Err(EngineError::backend_failure(
                        "scalar sparse-MoE activation is non-finite",
                    ));
                }
                write_scratch_f64(scratch, self.hidden_dimension + intermediate, value)?;
            }
            for hidden in 0..self.hidden_dimension {
                check_execution_cancelled(is_cancelled)?;
                let row = hidden
                    .checked_mul(self.intermediate_dimension)
                    .ok_or_else(|| EngineError::resource_exhaustion("expert down row overflows"))?;
                let mut projected = 0.0_f64;
                for column in 0..self.intermediate_dimension {
                    let element = row.checked_add(column).ok_or_else(|| {
                        EngineError::resource_exhaustion("expert index overflows")
                    })?;
                    projected += f64::from(read_f32(bytes, down_offset, element)?)
                        * read_scratch_f64(scratch, self.hidden_dimension + column)?;
                }
                let combined = read_scratch_f64(scratch, hidden)?
                    + projected * f64::from(routed_expert.weight);
                if !combined.is_finite() {
                    return Err(EngineError::backend_failure(
                        "scalar sparse-MoE output is non-finite",
                    ));
                }
                write_scratch_f64(scratch, hidden, combined)?;
            }
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.hidden_dimension)
            .map_err(|_| EngineError::resource_exhaustion("MoE output allocation failed"))?;
        for hidden in 0..self.hidden_dimension {
            check_execution_cancelled(is_cancelled)?;
            output.push(f32_from_f64(
                read_scratch_f64(scratch, hidden)?,
                "scalar sparse-MoE output exceeds F32",
            )?);
        }
        Ok(output)
    }

    fn validate_input(&self, input: &[f32]) -> Result<(), EngineError> {
        if input.len() != self.hidden_dimension || input.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_input("scalar sparse-MoE input is invalid"));
        }
        Ok(())
    }

    fn scratch_bytes(&self) -> Result<usize, EngineError> {
        self.hidden_dimension
            .checked_add(self.intermediate_dimension)
            .and_then(|elements| elements.checked_mul(size_of::<f64>()))
            .ok_or_else(|| EngineError::resource_exhaustion("sparse-MoE scratch size overflows"))
    }
}

fn check_execution_cancelled(is_cancelled: &impl Fn() -> bool) -> Result<(), EngineError> {
    if is_cancelled() { Err(EngineError::cancellation(None)) } else { Ok(()) }
}

fn read_f32(bytes: &[u8], base: usize, element: usize) -> Result<f32, EngineError> {
    let offset = element
        .checked_mul(size_of::<f32>())
        .and_then(|value| base.checked_add(value))
        .ok_or_else(|| EngineError::resource_exhaustion("expert byte offset overflows"))?;
    let end = offset
        .checked_add(size_of::<f32>())
        .ok_or_else(|| EngineError::resource_exhaustion("expert byte range overflows"))?;
    let exact: [u8; 4] = bytes
        .get(offset..end)
        .ok_or_else(|| EngineError::corrupt_model_package("expert byte range is truncated"))?
        .try_into()
        .map_err(|_| EngineError::internal_invariant("expert F32 width changed"))?;
    let value = f32::from_le_bytes(exact);
    if !value.is_finite() {
        return Err(EngineError::corrupt_model_package(
            "expert payload contains a non-finite value",
        ));
    }
    Ok(value)
}

#[cfg(test)]
fn fallible_zeroed_bytes(length: usize, reason: &'static str) -> Result<Vec<u8>, EngineError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| EngineError::resource_exhaustion(reason))?;
    values.resize(length, 0);
    Ok(values)
}

fn read_scratch_f64(bytes: &[u8], element: usize) -> Result<f64, EngineError> {
    let offset = element
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| EngineError::resource_exhaustion("scratch byte offset overflows"))?;
    let end = offset
        .checked_add(size_of::<f64>())
        .ok_or_else(|| EngineError::resource_exhaustion("scratch byte range overflows"))?;
    let exact: [u8; 8] = bytes
        .get(offset..end)
        .ok_or_else(|| EngineError::internal_invariant("scratch byte range is truncated"))?
        .try_into()
        .map_err(|_| EngineError::internal_invariant("scratch F64 width changed"))?;
    Ok(f64::from_le_bytes(exact))
}

fn write_scratch_f64(bytes: &mut [u8], element: usize, value: f64) -> Result<(), EngineError> {
    let offset = element
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| EngineError::resource_exhaustion("scratch byte offset overflows"))?;
    let end = offset
        .checked_add(size_of::<f64>())
        .ok_or_else(|| EngineError::resource_exhaustion("scratch byte range overflows"))?;
    let destination = bytes
        .get_mut(offset..end)
        .ok_or_else(|| EngineError::internal_invariant("scratch byte range is truncated"))?;
    destination.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[allow(clippy::cast_possible_truncation)] // Explicit finite F32 bounds are checked first.
fn f32_from_f64(value: f64, reason: &'static str) -> Result<f32, EngineError> {
    if !value.is_finite() || value > f64::from(f32::MAX) || value < f64::from(f32::MIN) {
        return Err(EngineError::backend_failure(reason));
    }
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(EngineError::backend_failure(reason));
    }
    Ok(narrowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    #[test]
    fn router_is_stable_on_ties_and_returns_expert_order() {
        let routed = route_sparse_experts(&[1.0, 3.0, 3.0, -2.0], 2).unwrap();
        assert_eq!(routed.iter().map(|value| value.expert()).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(routed[0].weight().to_bits(), 0.5_f32.to_bits());
        assert_eq!(routed[1].weight().to_bits(), 0.5_f32.to_bits());

        let signed_zero = route_sparse_experts(&[-0.0, 0.0], 1).unwrap();
        assert_eq!(signed_zero[0].expert(), 0);
        assert_eq!(signed_zero[0].weight().to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn router_rejects_nonfinite_and_invalid_top_k() {
        assert!(route_sparse_experts(&[0.0, f32::NAN], 1).is_err());
        assert!(route_sparse_experts(&[0.0, f32::INFINITY], 1).is_err());
        assert!(route_sparse_experts(&[0.0], 0).is_err());
        assert!(route_sparse_experts(&[0.0], 2).is_err());
    }

    #[test]
    fn router_handles_extreme_finite_score_separation_stably() {
        let routed = route_sparse_experts(&[f32::MAX, f32::MIN], 2).unwrap();
        assert_eq!(routed[0].expert(), 0);
        assert_eq!(routed[0].weight().to_bits(), 1.0_f32.to_bits());
        assert_eq!(routed[1].expert(), 1);
        assert_eq!(routed[1].weight().to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn scalar_expert_path_is_hand_calculated_and_deterministic() {
        let layer = ScalarSparseMoeLayer::new(1, 1, 2, 1, vec![1.0, -1.0]).unwrap();
        // Expert 0: gate=1, up=2, down=3. Expert 1 is not selected.
        let payloads = [encode(&[1.0, 2.0, 3.0]), encode(&[-1.0, 9.0, 9.0])];
        let first = layer.execute_f32_payload(&[2.0], |id| Ok(&payloads[id as usize])).unwrap();
        let second = layer.execute_f32_payload(&[2.0], |id| Ok(&payloads[id as usize])).unwrap();
        let expected = (2.0_f64 / (1.0 + (-2.0_f64).exp())) * 4.0 * 3.0;
        assert!((f64::from(first[0]) - expected).abs() < 1.0e-5);
        assert_eq!(first, second);
    }

    #[test]
    fn malformed_payload_and_operation_excess_fail_closed() {
        let layer = ScalarSparseMoeLayer::new(1, 1, 1, 1, vec![1.0]).unwrap();
        assert!(layer.execute_f32_payload(&[1.0], |_id| Ok(&[0_u8; 4])).is_err());
        let nan_payload = encode(&[f32::NAN, 1.0, 1.0]);
        let infinite_payload = encode(&[1.0, f32::INFINITY, 1.0]);
        assert!(layer.execute_f32_payload(&[1.0], |_id| Ok(&nan_payload)).is_err());
        assert!(layer.execute_f32_payload(&[1.0], |_id| Ok(&infinite_payload)).is_err());
        let valid_payload = encode(&[1.0, 1.0, 1.0]);
        assert!(layer.execute_f32_payload(&[f32::NAN], |_id| Ok(&valid_payload)).is_err());
        assert!(layer.execute_f32_payload(&[f32::INFINITY], |_id| Ok(&valid_payload)).is_err());
        assert!(ScalarSparseMoeLayer::new(65_536, 65_536, 1, 1, vec![0.0; 65_536]).is_err());
    }
}
