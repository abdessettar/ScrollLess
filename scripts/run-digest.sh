#!/usr/bin/env bash
# Run one edition from any working directory (used by the systemd unit and
# suitable for cron).
#
# Usage: scripts/run-digest.sh <edition>    # uses config.<edition>.yaml

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <edition>" >&2
  exit 64
fi
EDITION="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="config.${EDITION}.yaml"
if [ ! -f "$CONFIG" ]; then
  echo "Missing config file: $PROJECT_ROOT/$CONFIG" >&2
  exit 66
fi

# systemd and cron start with a minimal PATH; add the usual uv locations.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/opt/homebrew/bin:${PATH:-/usr/bin:/bin}"

exec uv run --frozen scrollless --config "$CONFIG"
