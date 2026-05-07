#!/usr/bin/env bash
# check-deps.sh — Check for available dependency updates without applying them.
#
# Usage:
#   ./scripts/check-deps.sh              # Check all deps in rusty-stack/
#   ./scripts/check-deps.sh --verbose     # Show full API responses
#   ./scripts/check-deps.sh --lag N       # Set lag period in days (default: 7)
#   ./scripts/check-deps.sh --dir PATH    # Set project directory (default: rusty-stack/)
#
# Exits 0 if all deps are up to date, 1 if updates are available, 2 on error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CRATE_DIR="$PROJECT_DIR/rusty-stack"
LAG_DAYS=7
VERBOSE=0

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose|-v)
      VERBOSE=1
      shift
      ;;
    --lag)
      LAG_DAYS="${2:-7}"
      shift 2
      ;;
    --dir)
      CRATE_DIR="$2"
      shift 2
      ;;
    --help|-h)
      head -15 "$0" | tail -12
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

CARGO_TOML="$CRATE_DIR/Cargo.toml"
CARGO_LOCK="$CRATE_DIR/Cargo.lock"

if [[ ! -f "$CARGO_TOML" ]]; then
  echo "❌ Cargo.toml not found at $CARGO_TOML"
  exit 2
fi

# Use Cargo.lock from the crate dir; fall back to project root
if [[ ! -f "$CARGO_LOCK" ]]; then
  CARGO_LOCK="$PROJECT_DIR/Cargo.lock"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Dependency Update Check"
echo "  Crate:     $CRATE_DIR"
echo "  Lag:       $LAG_DAYS days"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Extract direct dependencies from [dependencies] section ──
deps=()
while IFS= read -r line; do
  # Simple form: name = "version"
  if echo "$line" | grep -qP '^\s*[a-zA-Z0-9_-]+\s*=\s*"[^"]+"'; then
    dep_name=$(echo "$line" | grep -oP '^\s*\K[a-zA-Z0-9_-]+')
    dep_ver=$(echo "$line" | grep -oP '"\K[^"]+')
    [[ -z "$dep_ver" ]] && continue
    deps+=("$dep_name")
  fi
done < <(sed -n '/^\[dependencies\]/,/^\[/{ /^\[dependencies\]/d; /^\[/d; p }' "$CARGO_TOML")

if [[ ${#deps[@]} -eq 0 ]]; then
  echo "No dependencies found in Cargo.toml"
  exit 0
fi

echo "Found ${#deps[@]} direct dependencies. Checking crates.io..."
echo ""

up_to_date=0
updates_available=0
lag_blocked=0
check_failed=0

for dep in "${deps[@]}"; do
  # Get current locked version
  locked_ver="0.0.0"
  if [[ -f "$CARGO_LOCK" ]]; then
    locked_ver=$(grep -A2 "name = \"$dep\"" "$CARGO_LOCK" 2>/dev/null | grep "version" | head -1 | grep -oP '"\K[^"]+' || echo "0.0.0")
  fi

  # Query crates.io
  api_response=$(curl -sf "https://crates.io/api/v1/crates/$dep" \
    -H "User-Agent: rusty-stack-dep-checker (github.com/scooter-lacroix)" 2>/dev/null) || {
    echo "  ⚠️  $dep — failed to query crates.io"
    ((check_failed++)) || true
    continue
  }

  latest_ver=$(echo "$api_response" | jq -r '.crate.max_stable_version // .crate.max_version // empty' 2>/dev/null) || continue
  publish_date=$(echo "$api_response" | jq -r '.crate.updated_at // empty' 2>/dev/null) || true

  [[ -z "$latest_ver" || "$latest_ver" == "null" ]] && continue

  if [[ "$VERBOSE" == "1" ]]; then
    echo "  API response for $dep: latest=$latest_ver, updated=$publish_date"
  fi

  # Check if there's a newer version
  if [[ "$locked_ver" == "$latest_ver" ]]; then
    echo "  ✅ $dep — $locked_ver (up to date)"
    ((up_to_date++)) || true
    continue
  fi

  # Use sort -V to compare
  if echo -e "$latest_ver\n$locked_ver" | sort -V -C 2>/dev/null; then
    echo "  ✅ $dep — $locked_ver (up to date)"
    ((up_to_date++)) || true
    continue
  fi

  # There is an update available — check lag period
  if [[ -n "$publish_date" && "$publish_date" != "null" ]]; then
    publish_epoch=$(date -d "$publish_date" +%s 2>/dev/null) || {
      echo "  📦 $dep — $locked_ver → $latest_ver (could not parse publish date)"
      ((updates_available++)) || true
      continue
    }
    lag_cutoff=$(date -d "$LAG_DAYS days ago" +%s)

    if [[ "$publish_epoch" -lt "$lag_cutoff" ]]; then
      days_since=$(( ( $(date +%s) - publish_epoch ) / 86400 ))
      echo "  📦 $dep — $locked_ver → $latest_ver (published $days_since days ago ✅)"
      ((updates_available++)) || true
    else
      days_since=$(( ( $(date +%s) - publish_epoch ) / 86400 ))
      echo "  ⏳ $dep — $locked_ver → $latest_ver (published $days_since days ago, lag: ${LAG_DAYS}d)"
      ((lag_blocked++)) || true
    fi
  else
    echo "  📦 $dep — $locked_ver → $latest_ver (no publish date, assuming eligible)"
    ((updates_available++)) || true
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Summary"
echo "───────────────────────────────────────────────────────────────"
echo "  Up to date:        $up_to_date"
echo "  Updates available: $updates_available"
echo "  Lag-blocked:       $lag_blocked"
echo "  Check failed:      $check_failed"
echo "═══════════════════════════════════════════════════════════════"

if [[ "$updates_available" -gt 0 ]]; then
  exit 1
elif [[ "$check_failed" -gt 0 ]]; then
  exit 2
else
  exit 0
fi
