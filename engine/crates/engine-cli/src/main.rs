//! Owner-facing CLI for diagnostics and the local CPU-reference service.

#[cfg(unix)]
mod os_signal;

use personalai_engine_crypto::{
    EngineResponseSigner, TrustedControlPlaneKey, TrustedControlPlaneKeys, TrustedSigner,
};
use personalai_engine_scheduler::SchedulerLimits;
#[cfg(unix)]
use personalai_engine_server::{
    ContinuousServeLimits, ContinuousServeSummary, ContinuousShutdown, CpuEngineRuntime,
    UnixFrameListener,
};
use personalai_engine_server::{
    EngineService, FOUNDATION_STATUS, ProtocolDescriptor, ServiceConfig,
};
use personalai_engine_types::{
    EngineError, EngineErrorKind, ModelId, ModelPackageHash, ResourceBudget, SignerKeyId,
};
use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(unix)]
use std::sync::Arc;
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(unix)]
use std::thread::{self, JoinHandle};
#[cfg(unix)]
use std::time::Duration;

const USAGE: &str = "usage:
  personalai-engine version
  personalai-engine capabilities
  personalai-engine check-config --model-root PATH --runtime-root PATH --socket PATH
  personalai-engine serve --model-root PATH --runtime-root PATH --socket PATH \
    --model-id ID --package BASENAME.paimodel --package-sha384 HEX96 \
    --signer-key-id HEX32 --signer-public-key HEX64 \
    --control-key-id HEX32 --control-public-key HEX64 \
    [--previous-control-key-id HEX32 --previous-control-public-key HEX64] \
    --response-key-id HEX32 \
    --max-memory-bytes U64 --max-kv-cache-bytes U64
  personalai-engine serve-continuous --model-root PATH --runtime-root PATH --socket PATH \
    --model-id ID --package BASENAME.paimodel --package-sha384 HEX96 \
    --signer-key-id HEX32 --signer-public-key HEX64 \
    --control-key-id HEX32 --control-public-key HEX64 \
    [--previous-control-key-id HEX32 --previous-control-public-key HEX64] \
    --response-key-id HEX32 \
    --max-memory-bytes U64 --max-kv-cache-bytes U64 \
    --max-active-sequences U32 --max-connections U32
  serving reads exactly 32 binary response-key seed bytes from stdin;
  continuous serving drains on SIGINT or SIGTERM";

#[derive(Debug)]
enum Command {
    Version,
    Capabilities,
    CheckConfig(ServiceConfig),
    Serve(Box<ServeConfig>),
}

#[derive(Debug)]
enum ServeMode {
    Single,
    #[cfg(unix)]
    Continuous(ContinuousServeLimits),
}

#[derive(Debug)]
struct ServeConfig {
    service: ServiceConfig,
    model_id: ModelId,
    package_name: PathBuf,
    package_hash: ModelPackageHash,
    signer_key_id: SignerKeyId,
    signer_public_key: [u8; 32],
    control_key_id: SignerKeyId,
    control_public_key: [u8; 32],
    previous_control_key: Option<(SignerKeyId, [u8; 32])>,
    response_key_id: SignerKeyId,
    budget: ResourceBudget,
    mode: ServeMode,
}

#[derive(Default)]
struct RawServeConfig {
    model_root: Option<PathBuf>,
    runtime_root: Option<PathBuf>,
    socket_path: Option<PathBuf>,
    model_id: Option<ModelId>,
    package_name: Option<PathBuf>,
    package_hash: Option<ModelPackageHash>,
    signer_key_id: Option<SignerKeyId>,
    signer_public_key: Option<[u8; 32]>,
    control_key_id: Option<SignerKeyId>,
    control_public_key: Option<[u8; 32]>,
    previous_control_key_id: Option<SignerKeyId>,
    previous_control_public_key: Option<[u8; 32]>,
    response_key_id: Option<SignerKeyId>,
    max_memory_bytes: Option<u64>,
    max_kv_cache_bytes: Option<u64>,
    max_active_sequences: Option<u32>,
    max_connections: Option<u32>,
}

#[derive(Debug)]
enum CliError {
    InvalidArguments(&'static str),
    Engine(EngineError),
    Io,
}

impl From<EngineError> for CliError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

fn main() -> ExitCode {
    let mut secrets = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut errors = io::stderr().lock();
    match run(std::env::args_os().skip(1), &mut secrets, &mut output, &mut errors) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::InvalidArguments(message)) => {
            let _write_result = writeln!(errors, "{message}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(CliError::Engine(error)) => {
            let _write_result = writeln!(errors, "configuration rejected: {error}");
            ExitCode::from(2)
        }
        Err(CliError::Io) => ExitCode::from(1),
    }
}

fn run(
    arguments: impl Iterator<Item = OsString>,
    secrets: &mut impl Read,
    output: &mut impl Write,
    errors: &mut impl Write,
) -> Result<(), CliError> {
    match parse_command(arguments)? {
        Command::Version => writeln!(
            output,
            "personalai-engine {} ({FOUNDATION_STATUS})",
            env!("CARGO_PKG_VERSION")
        )
        .map_err(|_| CliError::Io),
        Command::Capabilities => write_capabilities(output),
        Command::CheckConfig(config) => writeln!(
            output,
            "configuration valid; model_root={}; runtime_root={}; socket={}; listener_available={}",
            config.model_root().display(),
            config.runtime_root().display(),
            config.socket_path().display(),
            cfg!(unix),
        )
        .map_err(|_| CliError::Io),
        Command::Serve(config) => serve(*config, secrets, errors),
    }
}

fn parse_command(mut arguments: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let command = arguments.next().ok_or(CliError::InvalidArguments("command is required"))?;
    let command =
        command.to_str().ok_or(CliError::InvalidArguments("command must be valid UTF-8"))?;
    match command {
        "version" if arguments.next().is_none() => Ok(Command::Version),
        "capabilities" if arguments.next().is_none() => Ok(Command::Capabilities),
        "serve" => parse_serve(arguments, false).map(Box::new).map(Command::Serve),
        "serve-continuous" => parse_serve(arguments, true).map(Box::new).map(Command::Serve),
        "check-config" => parse_config(arguments).map(Command::CheckConfig),
        "version" | "capabilities" => {
            Err(CliError::InvalidArguments("selected command does not accept arguments"))
        }
        _ => Err(CliError::InvalidArguments("unknown command")),
    }
}

fn parse_serve(
    mut arguments: impl Iterator<Item = OsString>,
    continuous: bool,
) -> Result<ServeConfig, CliError> {
    let mut config = RawServeConfig::default();
    while let Some(flag) = arguments.next() {
        let flag =
            flag.to_str().ok_or(CliError::InvalidArguments("serve flag must be valid UTF-8"))?;
        let value =
            arguments.next().ok_or(CliError::InvalidArguments("serve flag requires a value"))?;
        let text =
            value.to_str().ok_or(CliError::InvalidArguments("serve value must be valid UTF-8"))?;
        match flag {
            "--model-root" if config.model_root.is_none() => {
                config.model_root = Some(PathBuf::from(value));
            }
            "--runtime-root" if config.runtime_root.is_none() => {
                config.runtime_root = Some(PathBuf::from(value));
            }
            "--socket" if config.socket_path.is_none() => {
                config.socket_path = Some(PathBuf::from(value));
            }
            "--model-id" if config.model_id.is_none() => {
                config.model_id = Some(ModelId::new(text)?);
            }
            "--package" if config.package_name.is_none() => {
                config.package_name = Some(PathBuf::from(value));
            }
            "--package-sha384" if config.package_hash.is_none() => {
                config.package_hash = Some(ModelPackageHash::from_sha384_hex(text)?);
            }
            "--signer-key-id" if config.signer_key_id.is_none() => {
                config.signer_key_id = Some(SignerKeyId::from_bytes(parse_hex(text)?)?);
            }
            "--signer-public-key" if config.signer_public_key.is_none() => {
                config.signer_public_key = Some(parse_hex(text)?);
            }
            "--control-key-id" if config.control_key_id.is_none() => {
                config.control_key_id = Some(SignerKeyId::from_bytes(parse_hex(text)?)?);
            }
            "--control-public-key" if config.control_public_key.is_none() => {
                config.control_public_key = Some(parse_hex(text)?);
            }
            "--previous-control-key-id" if config.previous_control_key_id.is_none() => {
                config.previous_control_key_id = Some(SignerKeyId::from_bytes(parse_hex(text)?)?);
            }
            "--previous-control-public-key" if config.previous_control_public_key.is_none() => {
                config.previous_control_public_key = Some(parse_hex(text)?);
            }
            "--response-key-id" if config.response_key_id.is_none() => {
                config.response_key_id = Some(SignerKeyId::from_bytes(parse_hex(text)?)?);
            }
            "--max-memory-bytes" if config.max_memory_bytes.is_none() => {
                config.max_memory_bytes = Some(parse_u64(text)?);
            }
            "--max-kv-cache-bytes" if config.max_kv_cache_bytes.is_none() => {
                config.max_kv_cache_bytes = Some(parse_u64(text)?);
            }
            "--max-active-sequences" if config.max_active_sequences.is_none() => {
                config.max_active_sequences = Some(parse_u32(text)?);
            }
            "--max-connections" if config.max_connections.is_none() => {
                config.max_connections = Some(parse_u32(text)?);
            }
            "--model-root"
            | "--runtime-root"
            | "--socket"
            | "--model-id"
            | "--package"
            | "--package-sha384"
            | "--signer-key-id"
            | "--signer-public-key"
            | "--control-key-id"
            | "--control-public-key"
            | "--previous-control-key-id"
            | "--previous-control-public-key"
            | "--response-key-id"
            | "--max-memory-bytes"
            | "--max-kv-cache-bytes"
            | "--max-active-sequences"
            | "--max-connections" => {
                return Err(CliError::InvalidArguments("serve flag may appear only once"));
            }
            _ => return Err(CliError::InvalidArguments("unknown serve flag")),
        }
    }
    finish_serve_config(config, continuous)
}

fn finish_serve_config(config: RawServeConfig, continuous: bool) -> Result<ServeConfig, CliError> {
    let service = ServiceConfig::new(
        required(config.model_root, "--model-root is required")?,
        required(config.runtime_root, "--runtime-root is required")?,
        required(config.socket_path, "--socket is required")?,
    )?;
    let (maximum_active_sequences, mode) =
        serve_mode(continuous, config.max_active_sequences, config.max_connections)?;
    let budget =
        serve_budget(config.max_memory_bytes, config.max_kv_cache_bytes, maximum_active_sequences)?;
    let previous_control_key =
        rotation_key_pair(config.previous_control_key_id, config.previous_control_public_key)?;
    Ok(ServeConfig {
        service,
        model_id: required(config.model_id, "--model-id is required")?,
        package_name: required(config.package_name, "--package is required")?,
        package_hash: required(config.package_hash, "--package-sha384 is required")?,
        signer_key_id: required(config.signer_key_id, "--signer-key-id is required")?,
        signer_public_key: required(config.signer_public_key, "--signer-public-key is required")?,
        control_key_id: required(config.control_key_id, "--control-key-id is required")?,
        control_public_key: required(
            config.control_public_key,
            "--control-public-key is required",
        )?,
        previous_control_key,
        response_key_id: required(config.response_key_id, "--response-key-id is required")?,
        budget,
        mode,
    })
}

fn serve_budget(
    max_memory_bytes: Option<u64>,
    max_kv_cache_bytes: Option<u64>,
    maximum_active_sequences: u32,
) -> Result<ResourceBudget, CliError> {
    ResourceBudget::new(
        required(max_memory_bytes, "--max-memory-bytes is required")?,
        required(max_kv_cache_bytes, "--max-kv-cache-bytes is required")?,
        maximum_active_sequences,
    )
    .map_err(CliError::Engine)
}

fn serve_mode(
    continuous: bool,
    max_active_sequences: Option<u32>,
    max_connections: Option<u32>,
) -> Result<(u32, ServeMode), CliError> {
    if !continuous {
        if max_active_sequences.is_some() || max_connections.is_some() {
            return Err(CliError::InvalidArguments(
                "continuous limits require the serve-continuous command",
            ));
        }
        return Ok((1, ServeMode::Single));
    }
    #[cfg(unix)]
    {
        let maximum_active_sequences =
            required(max_active_sequences, "--max-active-sequences is required")?;
        let maximum_connections = required(max_connections, "--max-connections is required")?;
        let limits = ContinuousServeLimits::new(maximum_connections)?;
        if maximum_active_sequences > limits.maximum_connections() {
            return Err(CliError::InvalidArguments(
                "active sequence limit cannot exceed connection limit",
            ));
        }
        Ok((maximum_active_sequences, ServeMode::Continuous(limits)))
    }
    #[cfg(not(unix))]
    {
        let _limits = (max_active_sequences, max_connections);
        Err(CliError::Engine(EngineError::not_implemented(
            "continuous native engine serving requires a Unix platform",
        )))
    }
}

fn rotation_key_pair(
    key_id: Option<SignerKeyId>,
    public_key: Option<[u8; 32]>,
) -> Result<Option<(SignerKeyId, [u8; 32])>, CliError> {
    match (key_id, public_key) {
        (None, None) => Ok(None),
        (Some(key_id), Some(public_key)) => Ok(Some((key_id, public_key))),
        _ => Err(CliError::InvalidArguments(
            "previous control key ID and public key must be configured together",
        )),
    }
}

fn required<T>(value: Option<T>, message: &'static str) -> Result<T, CliError> {
    value.ok_or(CliError::InvalidArguments(message))
}

fn parse_u64(value: &str) -> Result<u64, CliError> {
    value
        .parse()
        .map_err(|_| CliError::InvalidArguments("serve byte budget must be an unsigned integer"))
}

fn parse_u32(value: &str) -> Result<u32, CliError> {
    value.parse().map_err(|_| {
        CliError::InvalidArguments("serve count limit must be an unsigned 32-bit integer")
    })
}

fn parse_hex<const N: usize>(value: &str) -> Result<[u8; N], CliError> {
    if value.len() != N * 2 {
        return Err(CliError::InvalidArguments("hexadecimal value has the wrong length"));
    }
    let mut output = [0_u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])
            .ok_or(CliError::InvalidArguments("hexadecimal value contains invalid text"))?;
        let low = hex_nibble(pair[1])
            .ok_or(CliError::InvalidArguments("hexadecimal value contains invalid text"))?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

const fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[cfg(unix)]
fn serve(
    config: ServeConfig,
    secrets: &mut impl Read,
    errors: &mut impl Write,
) -> Result<(), CliError> {
    let signer = TrustedSigner::active(config.signer_key_id, config.signer_public_key)?;
    let control_key =
        TrustedControlPlaneKey::active(config.control_key_id, config.control_public_key)?;
    let mut control_keys = vec![control_key];
    if let Some((key_id, public_key)) = config.previous_control_key {
        control_keys.push(TrustedControlPlaneKey::active(key_id, public_key)?);
    }
    let control_keys = TrustedControlPlaneKeys::new(control_keys)?;
    let mut response_seed = [0_u8; 32];
    secrets.read_exact(&mut response_seed).map_err(|_| CliError::Io)?;
    let response_signer =
        EngineResponseSigner::from_seed(config.response_key_id, &mut response_seed)?;
    let mut service = EngineService::new(
        config.service,
        SchedulerLimits::new(128, config.budget.max_concurrent_sequences(), 64 << 20)?,
    )?;
    service.register_configured_model(
        config.model_id.clone(),
        config.package_name,
        config.package_hash,
        signer,
    )?;
    let mut runtime =
        CpuEngineRuntime::new_with_control_keys(service, control_keys, response_signer)?;
    runtime.load_registered_model(&config.model_id, &config.budget)?;
    let listener = UnixFrameListener::bind(runtime.service().config())?;
    match config.mode {
        ServeMode::Single => serve_single(&mut runtime, &listener, errors),
        ServeMode::Continuous(limits) => {
            let summary = serve_continuous(&mut runtime, &listener, limits)?;
            writeln!(
                errors,
                "continuous service drained; accepted={}; authenticated={}; completed={}; cancelled={}; failed={}; peak_owned={}",
                summary.accepted_connections(),
                summary.authenticated_commands(),
                summary.completed_generations(),
                summary.cancelled_commands(),
                summary.failed_connections(),
                summary.peak_owned_connections(),
            )
            .map_err(|_| CliError::Io)
        }
    }
}

#[cfg(unix)]
fn serve_single(
    runtime: &mut CpuEngineRuntime,
    listener: &UnixFrameListener,
    errors: &mut impl Write,
) -> Result<(), CliError> {
    loop {
        if let Err(error) = runtime.serve_one(listener) {
            match error.kind() {
                EngineErrorKind::InvalidInput
                | EngineErrorKind::Cancellation
                | EngineErrorKind::AuthenticationFailure
                | EngineErrorKind::ProtocolViolation => {
                    writeln!(errors, "local request rejected: {error}")
                        .map_err(|_| CliError::Io)?;
                }
                _ => return Err(CliError::Engine(error)),
            }
        }
    }
}

#[cfg(unix)]
fn serve_continuous(
    runtime: &mut CpuEngineRuntime,
    listener: &UnixFrameListener,
    limits: ContinuousServeLimits,
) -> Result<ContinuousServeSummary, CliError> {
    let signals = os_signal::ProcessSignalGuard::install()?;
    let shutdown = ContinuousShutdown::new();
    let watcher_stop = Arc::new(AtomicBool::new(false));
    let watcher = spawn_shutdown_watcher(shutdown.clone(), Arc::clone(&watcher_stop));
    let watcher = match watcher {
        Ok(watcher) => watcher,
        Err(error) => {
            signals.restore()?;
            return Err(CliError::Engine(error));
        }
    };

    let serve_result = runtime.serve_continuous(listener, limits, &shutdown);
    watcher_stop.store(true, Ordering::Release);
    let watcher_result = watcher
        .join()
        .map_err(|_| EngineError::internal_invariant("process shutdown watcher panicked"));
    let restore_result = signals.restore();
    let summary = serve_result?;
    watcher_result?;
    restore_result?;
    Ok(summary)
}

#[cfg(unix)]
fn spawn_shutdown_watcher(
    shutdown: ContinuousShutdown,
    stop: Arc<AtomicBool>,
) -> Result<JoinHandle<()>, EngineError> {
    thread::Builder::new()
        .name("personalai-process-shutdown".to_owned())
        .stack_size(64 * 1_024)
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                if os_signal::ProcessSignalGuard::shutdown_requested() {
                    shutdown.request();
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        })
        .map_err(|_| EngineError::backend_failure("process shutdown watcher creation failed"))
}

#[cfg(not(unix))]
fn serve(
    _config: ServeConfig,
    _secrets: &mut impl Read,
    _errors: &mut impl Write,
) -> Result<(), CliError> {
    Err(CliError::Engine(EngineError::not_implemented(
        "native engine serving requires a Unix platform",
    )))
}

fn parse_config(mut arguments: impl Iterator<Item = OsString>) -> Result<ServiceConfig, CliError> {
    let mut model_root = None;
    let mut runtime_root = None;
    let mut socket_path = None;
    while let Some(flag) = arguments.next() {
        let flag = flag
            .to_str()
            .ok_or(CliError::InvalidArguments("configuration flag must be valid UTF-8"))?;
        let value = arguments
            .next()
            .ok_or(CliError::InvalidArguments("configuration flag requires a value"))?;
        match flag {
            "--model-root" if model_root.is_none() => model_root = Some(PathBuf::from(value)),
            "--runtime-root" if runtime_root.is_none() => {
                runtime_root = Some(PathBuf::from(value));
            }
            "--socket" if socket_path.is_none() => socket_path = Some(PathBuf::from(value)),
            "--model-root" | "--runtime-root" | "--socket" => {
                return Err(CliError::InvalidArguments("configuration flag may appear only once"));
            }
            _ => return Err(CliError::InvalidArguments("unknown configuration flag")),
        }
    }
    ServiceConfig::new(
        model_root.ok_or(CliError::InvalidArguments("--model-root is required"))?,
        runtime_root.ok_or(CliError::InvalidArguments("--runtime-root is required"))?,
        socket_path.ok_or(CliError::InvalidArguments("--socket is required"))?,
    )
    .map_err(CliError::Engine)
}

fn write_capabilities(output: &mut impl Write) -> Result<(), CliError> {
    let protocol = ProtocolDescriptor::foundation();
    writeln!(
        output,
        "protocol={}.{} frame_max={} listener={}",
        protocol.major, protocol.minor, protocol.maximum_frame_bytes, protocol.listener_implemented
    )
    .and_then(|()| writeln!(output, "cpu-reference=deterministic-decoder-execution"))
    .and_then(|()| writeln!(output, "arm64=not-implemented"))
    .and_then(|()| writeln!(output, "metal=not-implemented"))
    .and_then(|()| writeln!(output, "cuda=not-implemented"))
    .and_then(|()| writeln!(output, "model-loading=signed-first-party-package"))
    .map_err(|_| CliError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn version_and_capabilities_are_read_only() {
        let mut output = Vec::new();
        let mut errors = Vec::new();
        run(
            [OsString::from("version")].into_iter(),
            &mut Cursor::new(Vec::<u8>::new()),
            &mut output,
            &mut errors,
        )
        .expect("version must work");
        assert!(
            String::from_utf8(output).expect("output must be UTF-8").contains(FOUNDATION_STATUS)
        );
        assert!(errors.is_empty());
    }

    #[test]
    fn check_config_validates_without_opening_socket() {
        let arguments = [
            "check-config",
            "--model-root",
            "/models",
            "--runtime-root",
            "/run/personalai-engine",
            "--socket",
            "/run/personalai-engine/engine.sock",
        ]
        .into_iter()
        .map(OsString::from);
        let mut output = Vec::new();
        let mut errors = Vec::new();
        run(arguments, &mut Cursor::new(Vec::<u8>::new()), &mut output, &mut errors)
            .expect("valid config must pass");
        assert!(
            String::from_utf8(output)
                .expect("output must be UTF-8")
                .contains("listener_available=true")
        );
    }

    #[test]
    fn serve_requires_explicit_owner_configuration() {
        let mut output = Vec::new();
        let mut errors = Vec::new();
        assert!(matches!(
            run(
                [OsString::from("serve")].into_iter(),
                &mut Cursor::new(Vec::<u8>::new()),
                &mut output,
                &mut errors,
            ),
            Err(CliError::InvalidArguments("--model-root is required"))
        ));
        assert!(errors.is_empty());
    }

    #[test]
    fn serve_previous_control_identity_is_an_atomic_pair() {
        let mut incomplete = complete_serve_arguments();
        incomplete.extend([
            OsString::from("--previous-control-key-id"),
            OsString::from("05050505050505050505050505050505"),
        ]);
        assert!(matches!(
            parse_command(incomplete.into_iter()),
            Err(CliError::InvalidArguments(
                "previous control key ID and public key must be configured together"
            ))
        ));

        let mut complete = complete_serve_arguments();
        complete.extend([
            OsString::from("--previous-control-key-id"),
            OsString::from("05050505050505050505050505050505"),
            OsString::from("--previous-control-public-key"),
            OsString::from("06".repeat(32)),
        ]);
        let Command::Serve(config) =
            parse_command(complete.into_iter()).expect("complete rotation pair must parse")
        else {
            panic!("serve command must parse as serve");
        };
        assert!(config.previous_control_key.is_some());
    }

    #[cfg(unix)]
    #[test]
    fn continuous_serve_requires_explicit_bounded_limits() {
        let mut missing = complete_serve_arguments();
        missing[0] = OsString::from("serve-continuous");
        assert!(matches!(
            parse_command(missing.into_iter()),
            Err(CliError::InvalidArguments("--max-active-sequences is required"))
        ));

        let mut complete = complete_serve_arguments();
        complete[0] = OsString::from("serve-continuous");
        complete.extend([
            OsString::from("--max-active-sequences"),
            OsString::from("2"),
            OsString::from("--max-connections"),
            OsString::from("4"),
        ]);
        let Command::Serve(config) =
            parse_command(complete.into_iter()).expect("continuous serve limits must parse")
        else {
            panic!("continuous serve must use the serving command");
        };
        assert_eq!(config.budget.max_concurrent_sequences(), 2);
        assert!(
            matches!(config.mode, ServeMode::Continuous(limits) if limits.maximum_connections() == 4)
        );
    }

    #[cfg(unix)]
    #[test]
    fn continuous_active_limit_cannot_exceed_owned_connections() {
        let mut arguments = complete_serve_arguments();
        arguments[0] = OsString::from("serve-continuous");
        arguments.extend([
            OsString::from("--max-active-sequences"),
            OsString::from("3"),
            OsString::from("--max-connections"),
            OsString::from("2"),
        ]);
        assert!(matches!(
            parse_command(arguments.into_iter()),
            Err(CliError::InvalidArguments("active sequence limit cannot exceed connection limit"))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn process_signal_bridge_requests_existing_graceful_shutdown() {
        let signals = os_signal::ProcessSignalGuard::install()
            .expect("test process signal handlers must install");
        let shutdown = ContinuousShutdown::new();
        let stop = Arc::new(AtomicBool::new(false));
        let watcher = spawn_shutdown_watcher(shutdown.clone(), Arc::clone(&stop))
            .expect("shutdown watcher must start");
        os_signal::request_shutdown_for_test();
        watcher.join().expect("shutdown watcher must stop");
        assert!(shutdown.is_requested());
        stop.store(true, Ordering::Release);
        signals.restore().expect("test process signal handlers must restore");
    }

    #[cfg(unix)]
    #[test]
    #[expect(
        clippy::zombie_processes,
        reason = "the bounded poll reaps success with try_wait and timeout with kill plus wait"
    )]
    fn operating_system_termination_signal_reaches_the_bridge() {
        use std::io::BufRead;
        use std::process::{Command, Stdio};
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let mut child = Command::new(std::env::current_exe().expect("test executable must exist"))
            .args(["--exact", "tests::signal_subprocess_helper", "--nocapture"])
            .env("PERSONALAI_SIGNAL_HELPER", "1")
            .stdout(Stdio::piped())
            .spawn()
            .expect("signal helper must start");
        let stdout = child.stdout.take().expect("signal helper stdout must exist");
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let ready_reader = thread::spawn(move || {
            let mut readiness_sent = false;
            for line in std::io::BufReader::new(stdout).lines() {
                match line {
                    Ok(line) if line.trim() == "signal-ready" && !readiness_sent => {
                        let _send_result = ready_sender.send(Ok(()));
                        readiness_sent = true;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        if !readiness_sent {
                            let _send_result = ready_sender.send(Err(error));
                        }
                        return;
                    }
                }
            }
            if !readiness_sent {
                let _send_result = ready_sender.send(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "signal helper exited before readiness",
                )));
            }
        });
        ready_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("signal helper must report readiness within its bound")
            .expect("signal helper readiness must read");

        let status = Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .expect("local termination signal command must run");
        assert!(status.success());
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().expect("signal helper state must read") {
                assert!(status.success(), "signal helper failed: {status}");
                break;
            }
            if Instant::now() >= deadline {
                child.kill().expect("timed-out signal helper must stop");
                let _status = child.wait().expect("timed-out signal helper must reap");
                panic!("signal helper did not stop within its bound");
            }
            thread::sleep(Duration::from_millis(10));
        }
        ready_reader.join().expect("signal helper readiness reader must join");
    }

    #[cfg(unix)]
    #[test]
    fn signal_subprocess_helper() {
        if std::env::var_os("PERSONALAI_SIGNAL_HELPER").is_none() {
            return;
        }
        let signals =
            os_signal::ProcessSignalGuard::install().expect("signal helper handlers must install");
        println!("signal-ready");
        io::stdout().flush().expect("signal helper readiness must flush");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !os_signal::ProcessSignalGuard::shutdown_requested() {
            assert!(std::time::Instant::now() < deadline, "termination signal was not observed");
            thread::sleep(Duration::from_millis(10));
        }
        signals.restore().expect("signal helper handlers must restore");
    }

    fn complete_serve_arguments() -> Vec<OsString> {
        vec![
            "serve".to_owned(),
            "--model-root".to_owned(),
            "/models".to_owned(),
            "--runtime-root".to_owned(),
            "/run/personalai-engine".to_owned(),
            "--socket".to_owned(),
            "/run/personalai-engine/engine.sock".to_owned(),
            "--model-id".to_owned(),
            "decoder-v1".to_owned(),
            "--package".to_owned(),
            "decoder-v1.paimodel".to_owned(),
            "--package-sha384".to_owned(),
            "03".repeat(48),
            "--signer-key-id".to_owned(),
            "01010101010101010101010101010101".to_owned(),
            "--signer-public-key".to_owned(),
            "02".repeat(32),
            "--control-key-id".to_owned(),
            "03030303030303030303030303030303".to_owned(),
            "--control-public-key".to_owned(),
            "04".repeat(32),
            "--response-key-id".to_owned(),
            "07070707070707070707070707070707".to_owned(),
            "--max-memory-bytes".to_owned(),
            "1048576".to_owned(),
            "--max-kv-cache-bytes".to_owned(),
            "524288".to_owned(),
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }
}
