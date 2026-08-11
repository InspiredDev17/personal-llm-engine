//! Owner-facing CLI for diagnostics and the local CPU-reference service.

use personalai_engine_crypto::{TrustedControlPlaneKey, TrustedSigner};
use personalai_engine_scheduler::SchedulerLimits;
#[cfg(unix)]
use personalai_engine_server::{CpuEngineRuntime, UnixFrameListener};
use personalai_engine_server::{
    EngineService, FOUNDATION_STATUS, ProtocolDescriptor, ServiceConfig,
};
use personalai_engine_types::{
    EngineError, EngineErrorKind, ModelId, ModelPackageHash, ResourceBudget, SignerKeyId,
};
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(unix)]
use std::sync::atomic::AtomicBool;

const USAGE: &str = "usage:
  personalai-engine version
  personalai-engine capabilities
  personalai-engine check-config --model-root PATH --runtime-root PATH --socket PATH
  personalai-engine serve --model-root PATH --runtime-root PATH --socket PATH \
    --model-id ID --package BASENAME.paimodel --package-sha384 HEX96 \
    --signer-key-id HEX32 --signer-public-key HEX64 \
    --control-key-id HEX32 --control-public-key HEX64 \
    --max-memory-bytes U64 --max-kv-cache-bytes U64";

#[derive(Debug)]
enum Command {
    Version,
    Capabilities,
    CheckConfig(ServiceConfig),
    Serve(Box<ServeConfig>),
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
    budget: ResourceBudget,
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
    let mut output = io::stdout().lock();
    let mut errors = io::stderr().lock();
    match run(std::env::args_os().skip(1), &mut output, &mut errors) {
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
        Command::Serve(config) => serve(*config, errors),
    }
}

fn parse_command(mut arguments: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let command = arguments.next().ok_or(CliError::InvalidArguments("command is required"))?;
    let command =
        command.to_str().ok_or(CliError::InvalidArguments("command must be valid UTF-8"))?;
    match command {
        "version" if arguments.next().is_none() => Ok(Command::Version),
        "capabilities" if arguments.next().is_none() => Ok(Command::Capabilities),
        "serve" => parse_serve(arguments).map(Box::new).map(Command::Serve),
        "check-config" => parse_config(arguments).map(Command::CheckConfig),
        "version" | "capabilities" => {
            Err(CliError::InvalidArguments("selected command does not accept arguments"))
        }
        _ => Err(CliError::InvalidArguments("unknown command")),
    }
}

fn parse_serve(mut arguments: impl Iterator<Item = OsString>) -> Result<ServeConfig, CliError> {
    let mut model_root = None;
    let mut runtime_root = None;
    let mut socket_path = None;
    let mut model_id = None;
    let mut package_name = None;
    let mut package_hash = None;
    let mut signer_key_id = None;
    let mut signer_public_key = None;
    let mut control_key_id = None;
    let mut control_public_key = None;
    let mut max_memory_bytes = None;
    let mut max_kv_cache_bytes = None;
    while let Some(flag) = arguments.next() {
        let flag =
            flag.to_str().ok_or(CliError::InvalidArguments("serve flag must be valid UTF-8"))?;
        let value =
            arguments.next().ok_or(CliError::InvalidArguments("serve flag requires a value"))?;
        let text =
            value.to_str().ok_or(CliError::InvalidArguments("serve value must be valid UTF-8"))?;
        match flag {
            "--model-root" if model_root.is_none() => model_root = Some(PathBuf::from(value)),
            "--runtime-root" if runtime_root.is_none() => {
                runtime_root = Some(PathBuf::from(value));
            }
            "--socket" if socket_path.is_none() => socket_path = Some(PathBuf::from(value)),
            "--model-id" if model_id.is_none() => model_id = Some(ModelId::new(text)?),
            "--package" if package_name.is_none() => package_name = Some(PathBuf::from(value)),
            "--package-sha384" if package_hash.is_none() => {
                package_hash = Some(ModelPackageHash::from_sha384_hex(text)?);
            }
            "--signer-key-id" if signer_key_id.is_none() => {
                signer_key_id = Some(SignerKeyId::from_bytes(parse_hex(text)?)?);
            }
            "--signer-public-key" if signer_public_key.is_none() => {
                signer_public_key = Some(parse_hex(text)?);
            }
            "--control-key-id" if control_key_id.is_none() => {
                control_key_id = Some(SignerKeyId::from_bytes(parse_hex(text)?)?);
            }
            "--control-public-key" if control_public_key.is_none() => {
                control_public_key = Some(parse_hex(text)?);
            }
            "--max-memory-bytes" if max_memory_bytes.is_none() => {
                max_memory_bytes = Some(parse_u64(text)?);
            }
            "--max-kv-cache-bytes" if max_kv_cache_bytes.is_none() => {
                max_kv_cache_bytes = Some(parse_u64(text)?);
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
            | "--max-memory-bytes"
            | "--max-kv-cache-bytes" => {
                return Err(CliError::InvalidArguments("serve flag may appear only once"));
            }
            _ => return Err(CliError::InvalidArguments("unknown serve flag")),
        }
    }
    let service = ServiceConfig::new(
        required(model_root, "--model-root is required")?,
        required(runtime_root, "--runtime-root is required")?,
        required(socket_path, "--socket is required")?,
    )?;
    let budget = ResourceBudget::new(
        required(max_memory_bytes, "--max-memory-bytes is required")?,
        required(max_kv_cache_bytes, "--max-kv-cache-bytes is required")?,
        1,
    )?;
    Ok(ServeConfig {
        service,
        model_id: required(model_id, "--model-id is required")?,
        package_name: required(package_name, "--package is required")?,
        package_hash: required(package_hash, "--package-sha384 is required")?,
        signer_key_id: required(signer_key_id, "--signer-key-id is required")?,
        signer_public_key: required(signer_public_key, "--signer-public-key is required")?,
        control_key_id: required(control_key_id, "--control-key-id is required")?,
        control_public_key: required(control_public_key, "--control-public-key is required")?,
        budget,
    })
}

fn required<T>(value: Option<T>, message: &'static str) -> Result<T, CliError> {
    value.ok_or(CliError::InvalidArguments(message))
}

fn parse_u64(value: &str) -> Result<u64, CliError> {
    value
        .parse()
        .map_err(|_| CliError::InvalidArguments("serve byte budget must be an unsigned integer"))
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
fn serve(config: ServeConfig, errors: &mut impl Write) -> Result<(), CliError> {
    let signer = TrustedSigner::active(config.signer_key_id, config.signer_public_key)?;
    let control_key =
        TrustedControlPlaneKey::active(config.control_key_id, config.control_public_key)?;
    let mut service = EngineService::new(config.service, SchedulerLimits::new(128, 1, 64 << 20)?)?;
    service.register_configured_model(
        config.model_id.clone(),
        config.package_name,
        config.package_hash,
        signer,
    )?;
    let mut runtime = CpuEngineRuntime::new(service, control_key);
    runtime.load_registered_model(&config.model_id, &config.budget)?;
    let listener = UnixFrameListener::bind(runtime.service().config())?;
    let cancelled = AtomicBool::new(false);
    loop {
        if let Err(error) = runtime.serve_one(&listener, &cancelled) {
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

#[cfg(not(unix))]
fn serve(_config: ServeConfig, _errors: &mut impl Write) -> Result<(), CliError> {
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

    #[test]
    fn version_and_capabilities_are_read_only() {
        let mut output = Vec::new();
        let mut errors = Vec::new();
        run([OsString::from("version")].into_iter(), &mut output, &mut errors)
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
        run(arguments, &mut output, &mut errors).expect("valid config must pass");
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
            run([OsString::from("serve")].into_iter(), &mut output, &mut errors),
            Err(CliError::InvalidArguments("--model-root is required"))
        ));
        assert!(errors.is_empty());
    }
}
