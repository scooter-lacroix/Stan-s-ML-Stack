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
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CRATE_DIR="$PROJECT_DIR/rusty-stack"
LAG_DAYS=7
VERBOSE=0

for required_cmd in curl jq grep sed sort date; do
  if ! command -v "$required_cmd" >/dev/null 2>&1; then
    echo "❌ Missing required command: $required_cmd"
    exit 2
  fi
done

parse_iso_epoch() {
  local value="$1"
  if date -d "$value" +%s >/dev/null 2>&1; then
    date -d "$value" +%s
    return 0
  fi
  if date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$value" +%s >/dev/null 2>&1; then
    date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$value" +%s
    return 0
  fi
  return 1
}

days_ago_epoch() {
  local days="$1"
  if date -d "$days days ago" +%s >/dev/null 2>&1; then
    date -d "$days days ago" +%s
    return 0
  fi
  if date -v-"${days}"d +%s >/dev/null 2>&1; then
    date -v-"${days}"d +%s
    return 0
  fi
  return 1
}

SORT_HAS_VERSION=0
if printf '1.0.0\n1.0.0\n' | sort -V -C >/dev/null 2>&1; then
  SORT_HAS_VERSION=1
fi

normalize_version_core() {
  local value="${1#v}"
  # Ignore prerelease/build metadata for portable numeric fallback comparison.
  printf '%s\n' "${value%%[-+]*}"
}

version_ge_fallback() {
  local left right
  local -a left_parts right_parts
  local i max lp rp

  left="$(normalize_version_core "$1")"
  right="$(normalize_version_core "$2")"

  IFS='.' read -r -a left_parts <<< "$left"
  IFS='.' read -r -a right_parts <<< "$right"

  max=${#left_parts[@]}
  if (( ${#right_parts[@]} > max )); then
    max=${#right_parts[@]}
  fi

  for ((i=0; i<max; i++)); do
    lp="${left_parts[i]:-0}"
    rp="${right_parts[i]:-0}"
    lp="${lp%%[^0-9]*}"
    rp="${rp%%[^0-9]*}"
    [[ -z "$lp" ]] && lp=0
    [[ -z "$rp" ]] && rp=0

    if (( lp > rp )); then
      return 0
    fi
    if (( lp < rp )); then
      return 1
    fi
  done

  return 0
}

version_ge() {
  local locked="$1"
  local latest="$2"
  if [[ "$locked" == "$latest" ]]; then
    return 0
  fi

  if [[ "$SORT_HAS_VERSION" == "1" ]]; then
    if printf '%s\n%s\n' "$latest" "$locked" | sort -V -C >/dev/null 2>&1; then
      return 0
    fi
    return 1
  fi

  version_ge_fallback "$locked" "$latest"
}

collect_deps_from_metadata() {
  local manifest_path="$1"
  local metadata root_id

  metadata="$(cargo metadata --manifest-path "$manifest_path" --format-version 1 --no-deps 2>/dev/null)" || return 1
  root_id="$(printf '%s' "$metadata" | jq -r '.resolve.root // empty' 2>/dev/null)" || return 1
  [[ -z "$root_id" || "$root_id" == "null" ]] && return 1

  printf '%s' "$metadata" | jq -r --arg root "$root_id" '
    .packages[]
    | select(.id == $root)
    | .dependencies[]
    | select((.source // "") | startswith("registry+"))
    | .name
  ' 2>/dev/null | LC_ALL=C sort -u
}

collect_deps_from_manifest_legacy() {
  local manifest_path="$1"
  local line dep_name dep_ver

  # [dependencies] section
  while IFS= read -r line; do
    line="${line%%#*}"

    if echo "$line" | grep -Eq '^[[:space:]]*[a-zA-Z0-9_-]+[[:space:]]*=[[:space:]]*"[^"]+"'; then
      dep_name=$(echo "$line" | sed -E 's/^[[:space:]]*([a-zA-Z0-9_-]+)[[:space:]]*=.*/\1/')
      dep_ver=$(echo "$line" | sed -E 's/[^"]*"([^"]+)".*/\1/')
      [[ -n "$dep_ver" ]] && printf '%s\n' "$dep_name"
      continue
    fi

    if echo "$line" | grep -Eq '^[[:space:]]*[a-zA-Z0-9_-]+[[:space:]]*=[[:space:]]*\{[^}]*version[[:space:]]*=[[:space:]]*"[^"]+"'; then
      if echo "$line" | grep -Eq 'path[[:space:]]*=|git[[:space:]]*='; then
        continue
      fi
      dep_name=$(echo "$line" | sed -E 's/^[[:space:]]*([a-zA-Z0-9_-]+)[[:space:]]*=.*/\1/')
      dep_ver=$(echo "$line" | sed -E 's/.*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')
      [[ -n "$dep_ver" ]] && printf '%s\n' "$dep_name"
    fi
  done < <(sed -n '/^\[dependencies\]/,/^\[/{ /^\[dependencies\]/d; /^\[/d; p }' "$manifest_path")

  # [workspace.dependencies] section (workspace/root manifests)
  while IFS= read -r line; do
    line="${line%%#*}"
    if echo "$line" | grep -Eq '^[[:space:]]*[a-zA-Z0-9_-]+[[:space:]]*='; then
      dep_name=$(echo "$line" | sed -E 's/^[[:space:]]*([a-zA-Z0-9_-]+)[[:space:]]*=.*/\1/')
      [[ -n "$dep_name" ]] && printf '%s\n' "$dep_name"
    fi
  done < <(sed -n '/^\[workspace\.dependencies\]/,/^\[/{ /^\[workspace\.dependencies\]/d; /^\[/d; p }' "$manifest_path")
}

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

# ── Extract direct dependencies (prefer cargo metadata for workspace support) ──
deps=()
if command -v cargo >/dev/null 2>&1; then
  while IFS= read -r dep; do
    [[ -n "$dep" ]] && deps+=("$dep")
  done < <(collect_deps_from_metadata "$CARGO_TOML" || true)
fi

if [[ ${#deps[@]} -eq 0 ]]; then
  while IFS= read -r dep; do
    [[ -n "$dep" ]] && deps+=("$dep")
  done < <(collect_deps_from_manifest_legacy "$CARGO_TOML" | LC_ALL=C sort -u)
fi

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
    locked_ver=$(grep -A2 "name = \"$dep\"" "$CARGO_LOCK" 2>/dev/null | grep "version" | head -1 | sed -E 's/.*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/' || echo "0.0.0")
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

  # Version comparison (GNU sort -V when available, portable fallback otherwise)
  if version_ge "$locked_ver" "$latest_ver"; then
    echo "  ✅ $dep — $locked_ver (up to date)"
    ((up_to_date++)) || true
    continue
  fi

  # There is an update available — check lag period
  if [[ -n "$publish_date" && "$publish_date" != "null" ]]; then
    publish_epoch=$(parse_iso_epoch "$publish_date") || {
      echo "  📦 $dep — $locked_ver → $latest_ver (could not parse publish date)"
      ((updates_available++)) || true
      continue
    }
    lag_cutoff=$(days_ago_epoch "$LAG_DAYS") || {
      echo "  📦 $dep — $locked_ver → $latest_ver (could not compute lag cutoff)"
      ((updates_available++)) || true
      continue
    }

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
