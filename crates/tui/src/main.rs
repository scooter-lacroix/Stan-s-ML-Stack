//! ML Stack TUI Binary Entry Point
//!
//! Native Rust terminal UI for Stan's ML Stack installer.

use anyhow::Result;
use mlstack_tui::App;
use std::panic;

#[tokio::main]
async fn main() -> Result<()> {
    // Set up panic hook to restore terminal on crash
    panic::set_hook(Box::new(|info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let mut stdout = std::io::stdout();
        let _ = crossterm::execute!(
            stdout,
            crossterm::terminal::LeaveAlternateScreen
        );
        eprintln!("ML Stack TUI crashed: {info}");
    }));

    // Create and run the application
    let mut app = App::new()?;
    app.run().await?;

    Ok(())
}
