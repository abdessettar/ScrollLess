#!/usr/bin/env bash
# Install ScrollLess as systemd user timers, one per edition.
#
# Usage:
#   scripts/install-systemd.sh <edition>=<schedule> [<edition>=<schedule> ...]
#   scripts/install-systemd.sh --uninstall
#
# <edition> selects config.<edition>.yaml in the project root.
# <schedule> is a systemd OnCalendar expression, for example:
#   scripts/install-systemd.sh morning=08:00 evening=18:00
#   scripts/install-systemd.sh weekdays="Mon..Fri 07:30"
#
# Safe to re-run, for example after moving the project or changing a schedule.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATES="$PROJECT_ROOT/scripts/systemd"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

usage() {
  sed -n '4,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 64
}

uninstall() {
  local timers=()
  shopt -s nullglob
  for f in "$UNIT_DIR"/scrollless-*.timer; do
    timers+=("$(basename "$f")")
  done
  if [ "${#timers[@]}" -gt 0 ]; then
    systemctl --user disable --now "${timers[@]}"
  fi
  rm -f "$UNIT_DIR"/scrollless-*.timer "$UNIT_DIR/scrollless@.service"
  systemctl --user daemon-reload
  echo "Removed ScrollLess units from $UNIT_DIR"
}

[ "$#" -gt 0 ] || usage
if [ "$1" = "--uninstall" ]; then
  uninstall
  exit 0
fi

# Validate every argument before touching anything.
for arg in "$@"; do
  edition="${arg%%=*}"
  schedule="${arg#*=}"
  if [ "$edition" = "$arg" ] || [ -z "$schedule" ]; then
    echo "Expected <edition>=<schedule>, got: $arg" >&2
    usage
  fi
  if ! [[ "$edition" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid edition name: $edition (use letters, digits, - and _)" >&2
    exit 64
  fi
  if [ ! -f "$PROJECT_ROOT/config.$edition.yaml" ]; then
    echo "Missing config file: $PROJECT_ROOT/config.$edition.yaml" >&2
    exit 66
  fi
  if ! systemd-analyze calendar "$schedule" >/dev/null 2>&1; then
    echo "Invalid OnCalendar schedule for $edition: $schedule" >&2
    exit 64
  fi
done

mkdir -p "$UNIT_DIR"
sed "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" "$TEMPLATES/scrollless@.service" \
  >"$UNIT_DIR/scrollless@.service"
echo "Wrote $UNIT_DIR/scrollless@.service"

timers=()
for arg in "$@"; do
  edition="${arg%%=*}"
  schedule="${arg#*=}"
  sed -e "s|__EDITION__|$edition|g" -e "s|__SCHEDULE__|$schedule|g" \
    "$TEMPLATES/scrollless.timer.in" >"$UNIT_DIR/scrollless-$edition.timer"
  echo "Wrote $UNIT_DIR/scrollless-$edition.timer ($schedule)"
  timers+=("scrollless-$edition.timer")
done

systemctl --user daemon-reload
systemctl --user enable --now "${timers[@]}"

# Without lingering, user timers only fire while you are logged in.
if ! loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes'; then
  echo
  echo "Note: lingering is off, so timers only run while you are logged in."
  echo "Enable it with: sudo loginctl enable-linger $USER"
fi

echo
systemctl --user list-timers 'scrollless-*' --no-pager
