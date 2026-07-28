//! Rusty-Stack Library
//!
//! This library contains the core TUI installer, benchmark infrastructure,
//! and the new platform modules (core, platform, orchestrator, telemetry,
//! adapter).

pub mod adapter;
pub mod benchmark_logs;
pub mod benchmark_runners;
pub mod benchmarks;
pub mod bootstrap;
pub mod component_status;
pub mod config;
pub mod core;
pub mod gpu;
pub mod hardware;
pub mod installer;
pub mod installers;
pub mod logging;
pub mod orchestrator;
pub mod platform;
pub mod state;
pub mod telemetry;
pub mod uninstall;
pub mod verification;

/// Shared test infrastructure.
///
/// `env_lock()` is a SINGLE global mutex acquired by every test that mutates a
/// process-global env var (MLSTACK_USER_HOME, MLSTACK_PYTHON_BIN, ROCM_PATH,
/// package-manager overrides, DRY_RUN, GITHUB_TOKEN, …). Process env is global
/// state, so any two env-mutating tests running concurrently on different
/// threads corrupt each other — serializing ALL of them through this one lock
/// makes the suite deterministic under `cargo test`'s default thread pool.
#[cfg(test)]
pub mod test_support {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    static GLOBAL_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    /// The single global mutex serializing all env-mutating / global-state tests.
    pub fn env_lock() -> &'static Mutex<()> {
        GLOBAL_ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Acquire at the top of any test that mutates process-global state (env
    /// vars, the /tmp/llama-cpp-rocm-build dir, …). Returns a guard that serializes
    /// the test against every other such test.
    ///
    /// Recovers from **poison**: if a previous test panicked while holding the
    /// lock, the mutex is poisoned and a plain `.lock().unwrap()` would cascade
    /// the panic into every other locked test. We ignore the poison so each test
    /// fails (or passes) on its own merits, not as collateral damage.
    pub fn lock_env() -> MutexGuard<'static, ()> {
        env_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

#[cfg(feature = "tui")]
pub mod app;

#[cfg(feature = "tui")]
pub mod widgets;

use std::path::PathBuf;

/// Detect the scripts directory by checking current dir and parent dir.
pub fn detect_scripts_dir() -> String {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let scripts = cwd.join("scripts");
    if scripts.exists() {
        return scripts.to_string_lossy().to_string();
    }
    let parent_scripts = cwd.join("..").join("scripts");
    if parent_scripts.exists() {
        return parent_scripts.to_string_lossy().to_string();
    }
    "./scripts".to_string()
}

/// Set the MLSTACK_REPO_ROOT environment variable based on scripts dir location.
pub fn set_repo_root(scripts_dir: &str) {
    let scripts_path = PathBuf::from(scripts_dir);
    if let Some(root) = scripts_path.parent() {
        if root.join("stans_ml_stack").exists() {
            std::env::set_var("MLSTACK_REPO_ROOT", root);
        }
    }
}

/// Run the TUI application.
///
/// This function encapsulates the full TUI lifecycle: panic hook setup,
/// raw mode, alternate screen, terminal creation, app initialization,
/// event loop, and cleanup.
#[cfg(feature = "tui")]
pub fn run_tui() -> anyhow::Result<()> {
    use crate::app::App;
    use crossterm::execute;
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::Terminal;
    use std::io;
    use std::time::{Duration, Instant};

    std::panic::set_hook(Box::new(|info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, crossterm::cursor::Show);
        eprintln!("Rusty-Stack crashed: {info}");
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    let use_alt_screen = std::env::var("MLSTACK_NO_ALT_SCREEN").is_err();
    if use_alt_screen {
        execute!(stdout, EnterAlternateScreen)?;
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    // Hide the cursor for the duration of the TUI — it is never positioned
    // (no text-input cursor), so a visible cursor floats to the last-redrawn
    // cell each frame and shows up as a stray character. Restored on exit.
    //
    // If this fails, clean up the terminal state we already enabled (raw mode +
    // alternate screen) before propagating the error. The global panic hook
    // (above) only fires on unwinding, not on an early `?` return, so without
    // this cleanup a hide_cursor failure would leave the user's terminal in raw
    // mode inside the alternate screen — visibly broken.
    if let Err(e) = terminal.hide_cursor() {
        let _ = disable_raw_mode();
        if use_alt_screen {
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
        }
        return Err(e.into());
    }

    let scripts_dir = detect_scripts_dir();
    set_repo_root(&scripts_dir);
    let mut app = App::new(scripts_dir);

    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    let res = run_app(&mut terminal, &mut app, tick_rate, &mut last_tick);

    disable_raw_mode()?;
    if use_alt_screen {
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    }
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("Error: {err:?}");
    }

    Ok(())
}

/// Internal event loop for the TUI application.
#[cfg(feature = "tui")]
fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut ratatui::Terminal<B>,
    app: &mut crate::app::App,
    tick_rate: std::time::Duration,
    last_tick: &mut std::time::Instant,
) -> anyhow::Result<()>
where
    <B as ratatui::backend::Backend>::Error: Send + Sync + 'static,
{
    use crate::state::Stage;
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};

    loop {
        terminal.draw(|frame| app.draw(frame))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| std::time::Duration::from_secs(0));

        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) => {
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        return Ok(());
                    }
                    if key.code == KeyCode::Char('q') && app.stage == Stage::Recovery {
                        return Ok(());
                    }
                    app.handle_key(key);
                }
                Event::Resize(_, _) => {
                    terminal.autoresize()?;
                }
                _ => {}
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            *last_tick = std::time::Instant::now();
        }

        if app.should_exit {
            return Ok(());
        }
    }
}
