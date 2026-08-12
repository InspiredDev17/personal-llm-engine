//! Immutable allowlisted inference graph construction and execution.

use crate::operations::infer_broadcast_shape;
use crate::{CpuReferenceBackend, Device};
use personalai_engine_memory::MemoryPool;
use personalai_engine_tensor::{DType, Tensor, TensorShape};
use personalai_engine_types::{EngineError, MAX_RESOURCE_MEMORY_BYTES, MAX_TENSOR_COUNT};
use std::collections::{BTreeMap, BTreeSet};

/// Maximum inputs accepted by one first-party graph operation.
pub const MAX_NODE_INPUTS: usize = 4;

/// Stable identifier for a graph value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(u32);

impl ValueId {
    /// Creates a nonzero graph value identifier.
    pub fn new(value: u32) -> Result<Self, EngineError> {
        if value == 0 {
            Err(EngineError::invalid_input("graph value identifier must be nonzero"))
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the integer identifier.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable identifier for a graph node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphNodeId(u32);

impl GraphNodeId {
    /// Creates a nonzero graph node identifier.
    pub fn new(value: u32) -> Result<Self, EngineError> {
        if value == 0 {
            Err(EngineError::invalid_input("graph node identifier must be nonzero"))
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the integer identifier.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Static tensor type propagated while validating a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    dtype: DType,
    shape: TensorShape,
}

impl TensorSpec {
    /// Creates one validated static tensor type.
    pub fn new(dtype: DType, shape: TensorShape) -> Result<Self, EngineError> {
        shape.byte_len(dtype)?;
        Ok(Self { dtype, shape })
    }

    /// Returns the storage and execution data type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the tensor shape.
    pub const fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn byte_len(&self) -> Result<u64, EngineError> {
        self.shape.byte_len(self.dtype)
    }
}

/// Closed operation vocabulary implemented by first-party engine code.
#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    /// Canonical tensor copy.
    Copy,
    /// Explicit storage-type conversion.
    Convert(DType),
    /// Trailing-dimension broadcast addition.
    Add,
    /// Trailing-dimension broadcast multiplication.
    Multiply,
    /// Element-wise multiplication by a finite scalar.
    ScalarMultiply(f32),
    /// Rank-two matrix multiplied by rank-one vector.
    MatrixVector,
    /// Rank-two matrix multiplication.
    MatrixMatrix,
    /// Embedding row selection using a trusted token identifier.
    EmbeddingLookup(u32),
    /// RMS normalization across the final dimension.
    RmsNorm {
        /// Strictly positive normalization epsilon.
        epsilon: f32,
    },
    /// Stable softmax across the final dimension.
    Softmax,
    /// SiLU activation.
    Silu,
    /// SwiGLU activation over gate and up inputs.
    SwiGlu,
    /// Rotary positional encoding.
    Rope {
        /// Absolute sequence position.
        position: u32,
        /// Reviewed finite rotary frequency base greater than one.
        theta: f32,
    },
    /// Split a vector into heads or merge heads back into a vector.
    HeadReshape {
        /// Reviewed nonzero attention head count.
        head_count: u32,
    },
    /// Grouped-query causal attention.
    GroupedQueryAttention {
        /// Final key/value position visible to the query.
        causal_position: u32,
    },
    /// Causal masking of a rank-two score tensor.
    CausalMask {
        /// Absolute position represented by the first score row.
        first_query_position: u32,
    },
    /// Exact-shape residual addition.
    ResidualAdd,
    /// Hidden-to-vocabulary matrix-vector projection.
    LogitProjection,
    /// Sentinel accepted by construction APIs solely so validators can prove
    /// that unknown or future operation codes fail closed.
    Unsupported(u16),
}

/// One immutable graph node with explicit value edges.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphNode {
    id: GraphNodeId,
    operation: Operation,
    inputs: Box<[ValueId]>,
    output: ValueId,
}

impl GraphNode {
    /// Creates a bounded graph node.
    pub fn new(
        id: GraphNodeId,
        operation: Operation,
        inputs: impl Into<Box<[ValueId]>>,
        output: ValueId,
    ) -> Result<Self, EngineError> {
        let inputs = inputs.into();
        if inputs.len() > MAX_NODE_INPUTS {
            return Err(EngineError::invalid_input("graph node has too many inputs"));
        }
        Ok(Self { id, operation, inputs, output })
    }

    /// Returns the stable node identifier.
    pub const fn id(&self) -> GraphNodeId {
        self.id
    }

    /// Returns the allowlisted operation.
    pub const fn operation(&self) -> &Operation {
        &self.operation
    }

    /// Returns explicit input value identifiers.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    /// Returns the produced value identifier.
    pub const fn output(&self) -> ValueId {
        self.output
    }
}

/// Deterministic validated execution order and conservative storage bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    ordered_node_ids: Box<[GraphNodeId]>,
    temporary_storage_bytes: u64,
}

impl ExecutionPlan {
    /// Returns node identifiers in deterministic topological order.
    pub fn ordered_node_ids(&self) -> &[GraphNodeId] {
        &self.ordered_node_ids
    }

    /// Returns a conservative sum of all intermediate output bytes.
    pub const fn temporary_storage_bytes(&self) -> u64 {
        self.temporary_storage_bytes
    }
}

/// Validated immutable inference graph.
#[derive(Debug)]
pub struct InferenceGraph {
    inputs: BTreeMap<ValueId, TensorSpec>,
    nodes: Box<[GraphNode]>,
    outputs: Box<[ValueId]>,
    value_specs: BTreeMap<ValueId, TensorSpec>,
    plan: ExecutionPlan,
}

impl InferenceGraph {
    /// Validates and freezes a graph.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        inputs: impl IntoIterator<Item = (ValueId, TensorSpec)>,
        nodes: impl Into<Box<[GraphNode]>>,
        outputs: impl Into<Box<[ValueId]>>,
    ) -> Result<Self, EngineError> {
        let mut input_specs = BTreeMap::new();
        for (value, specification) in inputs {
            if input_specs.insert(value, specification).is_some() {
                return Err(EngineError::invalid_input("graph contains duplicate input values"));
            }
        }
        let nodes = nodes.into();
        let outputs = outputs.into();
        if nodes.len()
            > usize::try_from(MAX_TENSOR_COUNT).map_err(|_| {
                EngineError::resource_exhaustion("graph node bound is not addressable")
            })?
        {
            return Err(EngineError::resource_exhaustion("graph contains excessive nodes"));
        }
        if outputs.is_empty() {
            return Err(EngineError::invalid_input("graph requires at least one output"));
        }
        let unique_outputs = outputs.iter().copied().collect::<BTreeSet<_>>();
        if unique_outputs.len() != outputs.len() {
            return Err(EngineError::invalid_input("graph contains duplicate outputs"));
        }

        let mut node_ids = BTreeSet::new();
        let mut producers = BTreeMap::new();
        for (index, node) in nodes.iter().enumerate() {
            if !node_ids.insert(node.id) {
                return Err(EngineError::invalid_input(
                    "graph contains duplicate node identifiers",
                ));
            }
            if input_specs.contains_key(&node.output)
                || producers.insert(node.output, index).is_some()
            {
                return Err(EngineError::invalid_input("graph value has multiple producers"));
            }
            if matches!(node.operation, Operation::Unsupported(_)) {
                return Err(EngineError::not_implemented("graph operation is not allowlisted"));
            }
        }

        let mut indegrees = vec![0_usize; nodes.len()];
        let mut consumers = vec![Vec::<usize>::new(); nodes.len()];
        for (consumer_index, node) in nodes.iter().enumerate() {
            for input in &node.inputs {
                if input_specs.contains_key(input) {
                    continue;
                }
                let producer_index = producers.get(input).copied().ok_or_else(|| {
                    EngineError::invalid_input("graph node references a missing value")
                })?;
                indegrees[consumer_index] = indegrees[consumer_index]
                    .checked_add(1)
                    .ok_or_else(|| EngineError::resource_exhaustion("graph indegree overflows"))?;
                consumers[producer_index].push(consumer_index);
            }
        }

        let mut ready = BTreeSet::new();
        for (index, degree) in indegrees.iter().copied().enumerate() {
            if degree == 0 {
                ready.insert((nodes[index].id, index));
            }
        }
        let mut order = Vec::new();
        order.try_reserve_exact(nodes.len()).map_err(|_| {
            EngineError::resource_exhaustion("graph execution-order allocation failed")
        })?;
        while let Some(&(node_id, index)) = ready.first() {
            ready.remove(&(node_id, index));
            order.push(index);
            for consumer in &consumers[index] {
                indegrees[*consumer] = indegrees[*consumer]
                    .checked_sub(1)
                    .ok_or_else(|| EngineError::internal_invariant("graph indegree underflows"))?;
                if indegrees[*consumer] == 0 {
                    ready.insert((nodes[*consumer].id, *consumer));
                }
            }
        }
        if order.len() != nodes.len() {
            return Err(EngineError::invalid_input("graph contains a cycle"));
        }

        let mut value_specs = input_specs.clone();
        let mut temporary_storage_bytes = 0_u64;
        let mut ordered_node_ids = Vec::new();
        ordered_node_ids
            .try_reserve_exact(order.len())
            .map_err(|_| EngineError::resource_exhaustion("graph plan allocation failed"))?;
        for index in order {
            let node = &nodes[index];
            let mut specifications = Vec::new();
            specifications.try_reserve_exact(node.inputs.len()).map_err(|_| {
                EngineError::resource_exhaustion("graph shape validation allocation failed")
            })?;
            for input in &node.inputs {
                specifications.push(value_specs.get(input).ok_or_else(|| {
                    EngineError::internal_invariant("topological graph input is unavailable")
                })?);
            }
            let output_spec = infer_output(&node.operation, &specifications)?;
            temporary_storage_bytes = temporary_storage_bytes
                .checked_add(output_spec.byte_len()?)
                .ok_or_else(|| EngineError::resource_exhaustion("graph storage bound overflows"))?;
            if temporary_storage_bytes > MAX_RESOURCE_MEMORY_BYTES {
                return Err(EngineError::resource_exhaustion(
                    "graph temporary storage exceeds the hard limit",
                ));
            }
            value_specs.insert(node.output, output_spec);
            ordered_node_ids.push(node.id);
        }
        for output in &outputs {
            if !value_specs.contains_key(output) {
                return Err(EngineError::invalid_input("graph output value is missing"));
            }
        }
        let plan = ExecutionPlan {
            ordered_node_ids: ordered_node_ids.into_boxed_slice(),
            temporary_storage_bytes,
        };
        Ok(Self { inputs: input_specs, nodes, outputs, value_specs, plan })
    }

    /// Returns the validated static input map.
    pub const fn inputs(&self) -> &BTreeMap<ValueId, TensorSpec> {
        &self.inputs
    }

    /// Returns nodes in their original declaration order.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Returns requested graph output identifiers.
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    /// Returns the propagated type for any graph value.
    pub fn value_spec(&self, value: ValueId) -> Option<&TensorSpec> {
        self.value_specs.get(&value)
    }

    /// Returns the immutable deterministic execution plan.
    pub const fn execution_plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Executes the graph using tensors accounted by one aggregate pool.
    pub fn execute<B: Backend>(
        &self,
        backend: &B,
        mut values: BTreeMap<ValueId, Tensor>,
        pool: &MemoryPool,
    ) -> Result<BTreeMap<ValueId, Tensor>, EngineError> {
        if values.len() != self.inputs.len() {
            return Err(EngineError::invalid_input("graph runtime input set is incomplete"));
        }
        for (value, specification) in &self.inputs {
            let tensor = values
                .get(value)
                .ok_or_else(|| EngineError::invalid_input("graph runtime input is missing"))?;
            validate_runtime_tensor(tensor, specification, pool)?;
        }
        let nodes_by_id: BTreeMap<_, _> = self.nodes.iter().map(|node| (node.id, node)).collect();
        for node_id in self.plan.ordered_node_ids() {
            let node = nodes_by_id.get(node_id).ok_or_else(|| {
                EngineError::internal_invariant("execution plan references a missing node")
            })?;
            let output = {
                let mut inputs = Vec::<&Tensor>::new();
                inputs.try_reserve_exact(node.inputs.len()).map_err(|_| {
                    EngineError::resource_exhaustion("graph runtime edge allocation failed")
                })?;
                for input in &node.inputs {
                    inputs.push(values.get(input).ok_or_else(|| {
                        EngineError::internal_invariant("graph runtime value is unavailable")
                    })?);
                }
                backend.execute_operation(&node.operation, &inputs, pool)?
            };
            let specification = self.value_specs.get(&node.output).ok_or_else(|| {
                EngineError::internal_invariant("graph runtime output type is missing")
            })?;
            validate_runtime_tensor(&output, specification, pool)?;
            values.insert(node.output, output);
        }

        let mut outputs = BTreeMap::new();
        for output in &self.outputs {
            let tensor = values
                .remove(output)
                .ok_or_else(|| EngineError::internal_invariant("graph output was not produced"))?;
            outputs.insert(*output, tensor);
        }
        Ok(outputs)
    }
}

/// Closed backend execution boundary for validated operations.
pub trait Backend {
    /// Returns the trusted execution device.
    fn device(&self) -> Device;

    /// Executes one already-validated allowlisted operation.
    fn execute_operation(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError>;
}

impl Backend for CpuReferenceBackend {
    fn device(&self) -> Device {
        self.device()
    }

    fn execute_operation(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
        pool: &MemoryPool,
    ) -> Result<Tensor, EngineError> {
        let required = operation_input_count(operation)?;
        if inputs.len() != required {
            return Err(EngineError::invalid_input("graph operation input count is invalid"));
        }
        match operation {
            Operation::Copy => self.copy(inputs[0], pool),
            Operation::Convert(dtype) => self.convert(inputs[0], *dtype, pool),
            Operation::Add => self.add(inputs[0], inputs[1], pool),
            Operation::Multiply => self.multiply(inputs[0], inputs[1], pool),
            Operation::ScalarMultiply(scalar) => self.scalar_multiply(inputs[0], *scalar, pool),
            Operation::MatrixVector => self.matrix_vector(inputs[0], inputs[1], pool),
            Operation::MatrixMatrix => self.matrix_matrix(inputs[0], inputs[1], pool),
            Operation::EmbeddingLookup(token) => self.embedding_lookup(inputs[0], *token, pool),
            Operation::RmsNorm { epsilon } => self.rms_norm(inputs[0], inputs[1], *epsilon, pool),
            Operation::Softmax => self.softmax(inputs[0], pool),
            Operation::Silu => self.silu(inputs[0], pool),
            Operation::SwiGlu => self.swiglu(inputs[0], inputs[1], pool),
            Operation::Rope { position, theta } => self.rope(inputs[0], *position, *theta, pool),
            Operation::HeadReshape { head_count } => {
                self.head_reshape(inputs[0], *head_count, pool)
            }
            Operation::GroupedQueryAttention { causal_position } => self.grouped_query_attention(
                inputs[0],
                inputs[1],
                inputs[2],
                *causal_position,
                pool,
            ),
            Operation::CausalMask { first_query_position } => {
                self.causal_mask(inputs[0], *first_query_position, pool)
            }
            Operation::ResidualAdd => self.residual_add(inputs[0], inputs[1], pool),
            Operation::LogitProjection => self.logit_projection(inputs[0], inputs[1], pool),
            Operation::Unsupported(_) => {
                Err(EngineError::not_implemented("graph operation is not allowlisted"))
            }
        }
    }
}

fn operation_input_count(operation: &Operation) -> Result<usize, EngineError> {
    match operation {
        Operation::Copy
        | Operation::Convert(_)
        | Operation::ScalarMultiply(_)
        | Operation::EmbeddingLookup(_)
        | Operation::Softmax
        | Operation::Silu
        | Operation::Rope { .. }
        | Operation::HeadReshape { .. }
        | Operation::CausalMask { .. } => Ok(1),
        Operation::Add
        | Operation::Multiply
        | Operation::MatrixVector
        | Operation::MatrixMatrix
        | Operation::RmsNorm { .. }
        | Operation::SwiGlu
        | Operation::ResidualAdd
        | Operation::LogitProjection => Ok(2),
        Operation::GroupedQueryAttention { .. } => Ok(3),
        Operation::Unsupported(_) => {
            Err(EngineError::not_implemented("graph operation is not allowlisted"))
        }
    }
}

#[allow(clippy::too_many_lines)]
fn infer_output(operation: &Operation, inputs: &[&TensorSpec]) -> Result<TensorSpec, EngineError> {
    if inputs.len() != operation_input_count(operation)? {
        return Err(EngineError::invalid_input("graph operation input count is invalid"));
    }
    match operation {
        Operation::Copy => Ok(inputs[0].clone()),
        Operation::Convert(dtype) => {
            require_convert_dtype(inputs[0].dtype)?;
            require_convert_dtype(*dtype)?;
            TensorSpec::new(*dtype, inputs[0].shape.clone())
        }
        Operation::Add | Operation::Multiply => {
            require_f32_specs(inputs)?;
            TensorSpec::new(DType::F32, infer_broadcast_shape(&inputs[0].shape, &inputs[1].shape)?)
        }
        Operation::ScalarMultiply(scalar) => {
            require_f32_specs(inputs)?;
            require_finite_positive_or_signed(*scalar, "scalar multiply value is invalid", false)?;
            Ok(inputs[0].clone())
        }
        Operation::MatrixVector | Operation::LogitProjection => {
            require_f32_specs(inputs)?;
            if inputs[0].shape.rank() != 2
                || inputs[1].shape.rank() != 1
                || inputs[0].shape.dimensions()[1] != inputs[1].shape.dimensions()[0]
            {
                return Err(EngineError::invalid_input("matrix-vector graph shapes are invalid"));
            }
            TensorSpec::new(DType::F32, TensorShape::new([inputs[0].shape.dimensions()[0]])?)
        }
        Operation::MatrixMatrix => {
            require_f32_specs(inputs)?;
            if inputs[0].shape.rank() != 2
                || inputs[1].shape.rank() != 2
                || inputs[0].shape.dimensions()[1] != inputs[1].shape.dimensions()[0]
            {
                return Err(EngineError::invalid_input("matrix graph shapes are invalid"));
            }
            TensorSpec::new(
                DType::F32,
                TensorShape::new([
                    inputs[0].shape.dimensions()[0],
                    inputs[1].shape.dimensions()[1],
                ])?,
            )
        }
        Operation::EmbeddingLookup(token) => {
            require_f32_specs(inputs)?;
            if inputs[0].shape.rank() != 2 || u64::from(*token) >= inputs[0].shape.dimensions()[0] {
                return Err(EngineError::invalid_input("embedding graph metadata is invalid"));
            }
            TensorSpec::new(DType::F32, TensorShape::new([inputs[0].shape.dimensions()[1]])?)
        }
        Operation::RmsNorm { epsilon } => {
            require_f32_specs(inputs)?;
            if inputs[0].shape.rank() == 0
                || inputs[1].shape.rank() != 1
                || inputs[1].shape.dimensions()[0]
                    != *inputs[0].shape.dimensions().last().ok_or_else(|| {
                        EngineError::invalid_input("RMSNorm graph input rank is invalid")
                    })?
            {
                return Err(EngineError::invalid_input("RMSNorm graph shapes are invalid"));
            }
            require_finite_positive_or_signed(*epsilon, "RMSNorm epsilon is invalid", true)?;
            Ok(inputs[0].clone())
        }
        Operation::Softmax | Operation::Silu | Operation::CausalMask { .. } => {
            require_f32_specs(inputs)?;
            if matches!(operation, Operation::Softmax) && inputs[0].shape.rank() == 0 {
                return Err(EngineError::invalid_input("softmax graph input cannot be scalar"));
            }
            if matches!(operation, Operation::CausalMask { .. }) && inputs[0].shape.rank() != 2 {
                return Err(EngineError::invalid_input("causal mask graph input must be rank two"));
            }
            Ok(inputs[0].clone())
        }
        Operation::SwiGlu | Operation::ResidualAdd => {
            require_f32_specs(inputs)?;
            if inputs[0].shape != inputs[1].shape {
                return Err(EngineError::invalid_input("exact graph input shapes differ"));
            }
            Ok(inputs[0].clone())
        }
        Operation::Rope { theta, .. } => {
            require_f32_specs(inputs)?;
            if inputs[0].shape.rank() != 2 || inputs[0].shape.dimensions()[1] % 2 != 0 {
                return Err(EngineError::invalid_input("RoPE graph shape is invalid"));
            }
            require_finite_positive_or_signed(*theta, "RoPE theta is invalid", true)?;
            if *theta <= 1.0 {
                return Err(EngineError::invalid_input("RoPE theta is invalid"));
            }
            Ok(inputs[0].clone())
        }
        Operation::HeadReshape { head_count } => {
            require_f32_specs(inputs)?;
            if *head_count == 0 {
                return Err(EngineError::invalid_input("attention head count is zero"));
            }
            match inputs[0].shape.rank() {
                1 if inputs[0].shape.element_count() % u64::from(*head_count) == 0 => {
                    TensorSpec::new(
                        DType::F32,
                        TensorShape::new([
                            u64::from(*head_count),
                            inputs[0].shape.element_count() / u64::from(*head_count),
                        ])?,
                    )
                }
                2 if inputs[0].shape.dimensions()[0] == u64::from(*head_count) => TensorSpec::new(
                    DType::F32,
                    TensorShape::new([inputs[0].shape.element_count()])?,
                ),
                _ => Err(EngineError::invalid_input("attention head reshape is invalid")),
            }
        }
        Operation::GroupedQueryAttention { causal_position } => {
            require_f32_specs(inputs)?;
            let query = &inputs[0].shape;
            let key = &inputs[1].shape;
            if query.rank() != 2
                || key.rank() != 3
                || inputs[2].shape != *key
                || query.dimensions()[1] != key.dimensions()[2]
                || query.dimensions()[0] % key.dimensions()[1] != 0
                || u64::from(*causal_position) >= key.dimensions()[0]
            {
                return Err(EngineError::invalid_input("attention graph shapes are invalid"));
            }
            Ok(inputs[0].clone())
        }
        Operation::Unsupported(_) => {
            Err(EngineError::not_implemented("graph operation is not allowlisted"))
        }
    }
}

fn require_convert_dtype(dtype: DType) -> Result<(), EngineError> {
    if matches!(dtype, DType::F32 | DType::F16 | DType::BFloat16) {
        Ok(())
    } else {
        Err(EngineError::not_implemented("graph conversion data type is unsupported"))
    }
}

fn require_f32_specs(inputs: &[&TensorSpec]) -> Result<(), EngineError> {
    if inputs.iter().all(|input| input.dtype == DType::F32) {
        Ok(())
    } else {
        Err(EngineError::not_implemented("graph arithmetic requires f32 execution tensors"))
    }
}

fn require_finite_positive_or_signed(
    value: f32,
    detail: &'static str,
    positive: bool,
) -> Result<(), EngineError> {
    if value.is_finite() && (!positive || value > 0.0) {
        Ok(())
    } else {
        Err(EngineError::invalid_input(detail))
    }
}

fn validate_runtime_tensor(
    tensor: &Tensor,
    specification: &TensorSpec,
    pool: &MemoryPool,
) -> Result<(), EngineError> {
    if tensor.dtype() != specification.dtype || tensor.shape() != &specification.shape {
        return Err(EngineError::invalid_input("graph runtime tensor type does not match plan"));
    }
    if !tensor.accounted_by(pool) {
        return Err(EngineError::invalid_input(
            "graph runtime tensor uses a different memory pool",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use personalai_engine_tensor::MutableTensor;

    fn value(value: u32) -> ValueId {
        ValueId::new(value).unwrap()
    }

    fn node(value: u32) -> GraphNodeId {
        GraphNodeId::new(value).unwrap()
    }

    fn spec(shape: impl IntoIterator<Item = u64>) -> TensorSpec {
        TensorSpec::new(DType::F32, TensorShape::new(shape).unwrap()).unwrap()
    }

    #[test]
    fn deterministic_topology_and_full_execution_work() {
        let graph = InferenceGraph::new(
            [(value(1), spec([2])), (value(2), spec([2]))],
            vec![
                GraphNode::new(node(20), Operation::ScalarMultiply(2.0), [value(3)], value(4))
                    .unwrap(),
                GraphNode::new(node(10), Operation::Add, [value(1), value(2)], value(3)).unwrap(),
            ],
            vec![value(4)],
        )
        .unwrap();
        assert_eq!(graph.execution_plan().ordered_node_ids(), [node(10), node(20)]);
        assert_eq!(graph.execution_plan().temporary_storage_bytes(), 16);

        let pool = MemoryPool::new(1 << 20).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        let left =
            MutableTensor::from_f32_values(TensorShape::new([2]).unwrap(), &[1.0, 2.0], &pool)
                .unwrap();
        let right =
            MutableTensor::from_f32_values(TensorShape::new([2]).unwrap(), &[3.0, 4.0], &pool)
                .unwrap();
        let output = graph
            .execute(&backend, BTreeMap::from([(value(1), left), (value(2), right)]), &pool)
            .unwrap();
        assert_eq!(output[&value(4)].f32_value(0).unwrap().to_bits(), 8.0_f32.to_bits());
        assert_eq!(output[&value(4)].f32_value(1).unwrap().to_bits(), 12.0_f32.to_bits());
    }

    #[test]
    fn cycles_missing_values_duplicates_and_unsupported_ops_fail() {
        let cycle = vec![
            GraphNode::new(node(1), Operation::Copy, [value(3)], value(2)).unwrap(),
            GraphNode::new(node(2), Operation::Copy, [value(2)], value(3)).unwrap(),
        ];
        assert!(InferenceGraph::new([], cycle, vec![value(3)]).is_err());
        let missing = vec![GraphNode::new(node(1), Operation::Copy, [value(9)], value(2)).unwrap()];
        assert!(InferenceGraph::new([], missing, vec![value(2)]).is_err());
        let duplicate_nodes = vec![
            GraphNode::new(node(1), Operation::Copy, [value(1)], value(2)).unwrap(),
            GraphNode::new(node(1), Operation::Copy, [value(1)], value(3)).unwrap(),
        ];
        assert!(
            InferenceGraph::new([(value(1), spec([1]))], duplicate_nodes, vec![value(2)]).is_err()
        );
        let unsupported = vec![
            GraphNode::new(node(1), Operation::Unsupported(900), [value(1)], value(2)).unwrap(),
        ];
        assert!(InferenceGraph::new([(value(1), spec([1]))], unsupported, vec![value(2)]).is_err());
        assert!(
            InferenceGraph::new(
                [(value(1), spec([1]))],
                Vec::<GraphNode>::new(),
                vec![value(1), value(1)],
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_shape_propagation_and_runtime_inputs_fail() {
        let invalid = vec![
            GraphNode::new(node(1), Operation::MatrixVector, [value(1), value(2)], value(3))
                .unwrap(),
        ];
        assert!(
            InferenceGraph::new(
                [(value(1), spec([2, 3])), (value(2), spec([2]))],
                invalid,
                vec![value(3)],
            )
            .is_err()
        );

        let graph = InferenceGraph::new(
            [(value(1), spec([1]))],
            vec![GraphNode::new(node(1), Operation::Copy, [value(1)], value(2)).unwrap()],
            vec![value(2)],
        )
        .unwrap();
        let pool = MemoryPool::new(1024).unwrap();
        let backend = CpuReferenceBackend::new().unwrap();
        assert!(graph.execute(&backend, BTreeMap::new(), &pool).is_err());
    }
}
