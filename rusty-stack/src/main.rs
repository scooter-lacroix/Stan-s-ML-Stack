use crate::app::App;
use crate::state::Stage;
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

mod app;
mod component_status;
mod config;
mod hardware;
mod installer;
mod state;
mod widgets;

fn main() -> Result<()> {
    std::panic::set_hook(Box::new(|info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
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

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    tick_rate: Duration,
    last_tick: &mut Instant,
) -> Result<()> {
    loop {
        terminal.draw(|frame| app.draw(frame))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

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
            *last_tick = Instant::now();
        }

        if app.should_exit {
            return Ok(());
        }
    }
}

fn detect_scripts_dir() -> String {
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

fn set_repo_root(scripts_dir: &str) {
    let scripts_path = PathBuf::from(scripts_dir);
    if let Some(root) = scripts_path.parent() {
        if root.join("stans_ml_stack").exists() {
            std::env::set_var("MLSTACK_REPO_ROOT", root);
        }
    }
}
