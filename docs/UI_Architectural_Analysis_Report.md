# Rusty-Stack UI Architectural Notes

Rusty-Stack uses a Rust + ratatui + crossterm architecture with an explicit state machine.

## Stages

- Welcome
- Hardware Detect
- Preflight
- Component Select
- Configuration
- Confirm
- Installing
- Complete
- Recovery

## Data Flow

- Hardware + preflight run in background threads.
- Script installation runs with live stdout/stderr capture.
- Log output is persisted to `~/.mlstack/logs/rusty-stack.log`.

## Reliability

- `MLSTACK_NO_ALT_SCREEN=1` disables alternate screen usage.
- Panic hook restores terminal state on crash.
