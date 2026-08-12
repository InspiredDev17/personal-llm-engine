#!/bin/sh
set -eu

profile=${1:-ops/native-engine/macos/personalai-engine.sb}
case "$profile" in
  /*) ;;
  *) profile="$(pwd)/$profile" ;;
esac

test_root=$(mktemp -d "/tmp/paie-sandbox.XXXXXX")
test_root=$(cd "$test_root" && pwd -P)
runtime_root="$test_root/runtime"
model_root="$test_root/models"
socket_path="$runtime_root/engine.sock"
test_profile="$test_root/personalai-engine.sb"
tcp_port=$((42000 + ($$ % 1000)))
uds_pid=
tcp_pid=

cleanup() {
  if [ -n "$uds_pid" ]; then kill "$uds_pid" 2>/dev/null || true; fi
  if [ -n "$tcp_pid" ]; then kill "$tcp_pid" 2>/dev/null || true; fi
  wait "$uds_pid" 2>/dev/null || true
  wait "$tcp_pid" 2>/dev/null || true
  rm -rf "$test_root"
}
trap cleanup EXIT HUP INT TERM

mkdir -m 700 "$runtime_root" "$model_root"
/usr/bin/sed \
  "s#/private/var/run/personalai-engine/engine.sock#$socket_path#g" \
  "$profile" >"$test_profile"

/usr/bin/nc -lU "$socket_path" >/dev/null &
uds_pid=$!
for _attempt in 1 2 3 4 5 6 7 8 9 10; do
  [ -S "$socket_path" ] && break
  sleep 0.05
done
[ -S "$socket_path" ]

printf 'uds-proof\n' | /usr/bin/sandbox-exec \
  -D "MODEL_ROOT=$model_root" \
  -D "RUNTIME_ROOT=$runtime_root" \
  -D "SOCKET_PATH=$socket_path" \
  -f "$test_profile" /usr/bin/nc -U "$socket_path"

/usr/bin/nc -l 127.0.0.1 "$tcp_port" >/dev/null &
tcp_pid=$!
sleep 0.1
if /usr/bin/sandbox-exec \
  -D "MODEL_ROOT=$model_root" \
  -D "RUNTIME_ROOT=$runtime_root" \
  -D "SOCKET_PATH=$socket_path" \
  -f "$test_profile" /usr/bin/nc -z 127.0.0.1 "$tcp_port"; then
  echo "sandbox unexpectedly permitted loopback TCP" >&2
  exit 1
fi

echo "macOS sandbox proof passed: configured UDS allowed; loopback TCP denied"
