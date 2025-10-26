#!/bin/bash
# Stan's ML Stack - ROCm Channel Wrapper
# Provides a non-interactive interface to select ROCm channels when
# calling the primary install_rocm.sh script.

set -euo pipefail

CHANNEL="${1:-}" # optional positional argument

usage() {
    cat <<USAGE
Usage: $0 [CHANNEL]

Channels:
  legacy  - ROCm 6.4.3
  stable  - ROCm 7.0.0
  latest  - ROCm 7.0.2 (default)
  preview - ROCm 7.9.0 (technology preview)

You can also set the ROCM_CHANNEL environment variable. If neither
is provided, the script defaults to the "latest" channel.
USAGE
}

case "$CHANNEL" in
    "" )
        CHANNEL="${ROCM_CHANNEL:-latest}"
        ;;
    legacy|stable|latest|preview)
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown channel: $CHANNEL" >&2
        usage
        exit 1
        ;;
esac

if [ $# -gt 1 ]; then
    echo "Too many arguments" >&2
    usage
    exit 1
fi

case "$CHANNEL" in
    legacy)
        export INSTALL_ROCM_PRESEEDED_CHOICE=1
        ;;
    stable)
        export INSTALL_ROCM_PRESEEDED_CHOICE=2
        ;;
    latest)
        export INSTALL_ROCM_PRESEEDED_CHOICE=3
        ;;
    preview)
        export INSTALL_ROCM_PRESEEDED_CHOICE=4
        ;;
esac

export ROCM_CHANNEL="$CHANNEL"

dirname "${BASH_SOURCE[0]}" | {
    read -r SCRIPT_DIR
    SCRIPT_DIR=${SCRIPT_DIR:-.}
    "${SCRIPT_DIR}/install_rocm.sh"
}
