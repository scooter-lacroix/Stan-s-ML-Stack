//! TUI Application
//!
//! Main application state and ratatui integration with Rusty Stack branding.

use crate::components::{GPUPanel, LogPanel, StatusBar};
use crate::events::{Event, EventHandler};
use crate::screens::{MainScreen, Screen};
use anyhow::Result;
use chrono::Local;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use mlstack_hardware::{HardwareDiscovery, GPUFilter};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;
use std::io;
use std::io::Write;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

/// Terminal restoration guard that ensures terminal is properly restored on drop.
struct TerminalGuard {
    _stdout_exists: bool,  // Marker to ensure drop runs
}

impl TerminalGuard {
    fn new() -> Result<Self> {
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        // Flush to ensure commands are sent immediately
        let _ = stdout.flush();
        Ok(Self {
            _stdout_exists: true,
        })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        // Restore terminal state in the correct order
        let _ = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = crossterm::execute!(
            stdout,
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        // Flush to ensure all escape sequences are sent
        let _ = stdout.flush();
    }
}

/// Application stage (matching old Rusty Stack).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Welcome screen
    Welcome,
    /// Hardware detection
    HardwareDetect,
    /// Overview dashboard
    Overview,
    /// Component selection
    ComponentSelect,
    /// Installation
    Installing,
    /// Verification
    Verify,
    /// Settings
    Settings,
}

/// Application state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    /// Running normally
    Running,
    /// Should quit
    ShouldQuit,
    /// Installing components
    Installing,
    /// Verifying installation
    Verifying,
}

/// Hardware detection result
pub struct HardwareInfo {
    pub gpu_model: String,
    pub gpu_arch: String,
    pub gpu_memory_gb: f64,
    pub gpu_count: usize,
    pub rocm_version: String,
    pub os_name: String,
    pub kernel_version: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub available_memory_gb: f64,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self {
            gpu_model: "Detecting...".to_string(),
            gpu_arch: "Unknown".to_string(),
            gpu_memory_gb: 0.0,
            gpu_count: 0,
            rocm_version: "Not detected".to_string(),
            os_name: "Linux".to_string(),
            kernel_version: "Unknown".to_string(),
            cpu_model: "Unknown".to_string(),
            cpu_cores: 0,
            memory_gb: 0.0,
            available_memory_gb: 0.0,
        }
    }
}

/// Main TUI application.
pub struct App {
    /// Current application state
    pub state: AppState,
    /// Current stage
    pub stage: Stage,
    /// Current screen
    pub current_screen: Box<dyn Screen>,
    /// GPU panel
    pub gpu_panel: GPUPanel,
    /// Log panel
    pub log_panel: LogPanel,
    /// Status bar
    pub status_bar: StatusBar,
    /// Event handler
    event_handler: EventHandler,
    /// Tick count for animations
    tick_count: u64,
    /// Hardware info
    pub hardware: HardwareInfo,
    /// Hardware detection in progress
    hardware_receiver: Option<Receiver<HardwareInfo>>,
    /// Environment type
    pub env_type: String,
    /// Whether a redraw is needed (dirty flag)
    dirty: bool,
}

impl App {
    /// Creates a new TUI application.
    pub fn new() -> Result<Self> {
        let event_handler = EventHandler::new(Duration::from_millis(100));

        // Detect environment type
        let env_type = format!("{:?}", mlstack_env::EnvironmentManager::detect_environment());

        let mut app = Self {
            state: AppState::Running,
            stage: Stage::Welcome,
            current_screen: Box::new(MainScreen::new()),
            gpu_panel: GPUPanel::new(),
            log_panel: LogPanel::new(),
            status_bar: StatusBar::new(),
            event_handler,
            tick_count: 0,
            hardware: HardwareInfo::default(),
            hardware_receiver: None,
            env_type,
            dirty: true,
        };

        // Log startup message
        app.log("Rusty Stack TUI starting...");
        app.log(format!("Environment: {}", app.env_type));

        // Start hardware detection immediately
        app.start_hardware_detection();

        Ok(app)
    }

    /// Starts background hardware detection.
    fn start_hardware_detection(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.hardware_receiver = Some(rx);
        self.log("Starting hardware detection...");

        thread::spawn(move || {
            let discovery = HardwareDiscovery::new();
            let mut info = HardwareInfo::default();

            // Detect system info
            if let Ok(system) = discovery.detect_system() {
                info.os_name = system.os_version.clone();
                info.kernel_version = system.kernel_version.clone();
                info.cpu_model = system.cpu_model.clone();
                info.cpu_cores = system.cpu_cores;
                info.memory_gb = system.memory_gb();
                info.available_memory_gb = system.available_memory_gb();
            }

            // Detect GPUs
            if let Ok(gpus) = discovery.detect_gpus() {
                let discrete = GPUFilter::default().filter(gpus);
                if !discrete.is_empty() {
                    let gpu = &discrete[0];
                    info.gpu_model = gpu.model.clone();
                    info.gpu_arch = gpu.architecture.to_string();
                    info.gpu_memory_gb = gpu.memory_gb as f64;
                    info.gpu_count = discrete.len();
                }
            }

            // Detect ROCm
            if let Ok(rocm) = discovery.detect_rocm() {
                info.rocm_version = rocm.version;
            }

            let _ = tx.send(info);
        });
    }

    /// Polls for hardware detection results.
    fn poll_hardware(&mut self) {
        if let Some(rx) = &self.hardware_receiver {
            if let Ok(info) = rx.try_recv() {
                self.hardware = info;
                self.hardware_receiver = None;
                self.log("Hardware detection complete!");

                // Update GPU panel
                self.gpu_panel.update_gpu_info(vec![
                    format!("GPU: {}", self.hardware.gpu_model),
                    format!("Architecture: {}", self.hardware.gpu_arch),
                    format!("Memory: {:.1} GB", self.hardware.gpu_memory_gb),
                    format!("GPU Count: {}", self.hardware.gpu_count),
                    format!("ROCm: {}", self.hardware.rocm_version),
                ]);

                // Auto-advance to Overview if on Welcome/HardwareDetect
                if self.stage == Stage::Welcome || self.stage == Stage::HardwareDetect {
                    self.stage = Stage::Overview;
                }
            }
        }
    }

    /// Runs the application.
    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal with guard for automatic cleanup
        let _guard = TerminalGuard::new()?;
        let stdout = io::stdout();
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Main loop
        self.main_loop(&mut terminal).await
    }

    /// Main event loop.
    async fn main_loop<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        let mut last_tick = std::time::Instant::now();
        let tick_rate = Duration::from_millis(100);

        while self.state != AppState::ShouldQuit {
            // Only redraw if dirty or on animation tick
            if self.dirty {
                terminal.draw(|f| self.draw(f))?;
                self.dirty = false;
            }

            // Handle events
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if let Some(event) = self.event_handler.next_event(timeout).await {
                self.handle_event(event).await?;
                self.dirty = true; // Redraw after event
            }

            // Update animations (mark dirty on spinner change)
            if last_tick.elapsed() >= tick_rate {
                let old_tick = self.tick_count;
                self.update();
                // Only mark dirty if spinner frame changed (every 10 ticks)
                if self.tick_count / 10 != old_tick / 10 {
                    self.dirty = true;
                }
                last_tick = std::time::Instant::now();
            }
        }

        // Final draw to show clean exit state
        let _ = terminal.draw(|f| self.draw(f));

        Ok(())
    }

    /// Draws the UI with Rusty Stack branding.
    fn draw(&mut self, frame: &mut ratatui::Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),   // Header (Rusty Stack branding)
                Constraint::Min(10),     // Main content
                Constraint::Length(8),   // GPU panel
                Constraint::Length(8),   // Log panel
                Constraint::Length(1),   // Status bar
            ])
            .split(frame.size());

        // Draw header with Rusty Stack branding
        self.draw_header(frame, chunks[0]);

        // Draw main content based on stage/screen
        self.current_screen.draw(frame, chunks[1]);

        // Draw panels
        self.gpu_panel.draw(frame, chunks[2]);
        self.log_panel.draw(frame, chunks[3]);
        self.status_bar.draw(frame, chunks[4]);
    }

    /// Draws the branded header.
    fn draw_header(&self, frame: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let spinner = self.spinner();
        let title_lines = vec![
            Line::from(vec![
                Span::styled("✦", Style::default().fg(Color::Yellow)),
                Span::raw(" "),
                Span::styled(
                    "Rusty-Stack",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(spinner, Style::default().fg(Color::Magenta)),
            ]),
            Line::from(vec![
                Span::styled(
                    "AMD ML Stack Installer",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  •  "),
                Span::styled("Ratatui Edition", Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("Stage: {:?}", self.stage),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
        ];
        let title = Paragraph::new(Text::from(title_lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(
                        " Rusty Stack ",
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    ))
                    .border_style(Style::default().fg(Color::Cyan)),
            );
        frame.render_widget(title, area);
    }

    /// Gets spinner frame.
    fn spinner(&self) -> &'static str {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let idx = (self.tick_count % FRAMES.len() as u64) as usize;
        FRAMES[idx]
    }

    /// Handles events.
    async fn handle_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Key(key) => self.handle_key_event(key).await?,
            Event::Resize(_, _) => {
                // Terminal resized, will redraw automatically
            }
            Event::Tick => {
                // Animation tick
            }
        }
        Ok(())
    }

    /// Handles keyboard events.
    async fn handle_key_event(&mut self, key: crossterm::event::KeyEvent) -> Result<()> {
        use crossterm::event::{KeyCode, KeyModifiers};

        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.state = AppState::ShouldQuit;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.state = AppState::ShouldQuit;
            }
            KeyCode::Tab => {
                self.current_screen.next_tab();
            }
            KeyCode::BackTab => {
                self.current_screen.prev_tab();
            }
            KeyCode::Char('1') => {
                self.stage = Stage::Overview;
                self.current_screen.set_tab(0);
            }
            KeyCode::Char('2') => {
                self.stage = Stage::ComponentSelect;
                self.current_screen.set_tab(1);
            }
            KeyCode::Char('3') => {
                self.stage = Stage::Verify;
                self.current_screen.set_tab(2);
            }
            KeyCode::Char('4') => {
                self.stage = Stage::Settings;
                self.current_screen.set_tab(3);
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                // Refresh hardware detection
                self.log("Refreshing hardware detection...");
                self.start_hardware_detection();
            }
            _ => {
                // Pass to current screen
                self.current_screen.handle_key(key)?;
            }
        }
        Ok(())
    }

    /// Updates animations and state.
    fn update(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
        self.poll_hardware();
        self.current_screen.update();
        self.gpu_panel.update();
        self.log_panel.update();
    }

    /// Adds a log message.
    pub fn log(&mut self, message: impl Into<String>) {
        let timestamp = Local::now().format("%H:%M:%S");
        self.log_panel.add_message(format!("[{}] {}", timestamp, message.into()));
    }

    /// Sets the current screen.
    pub fn set_screen(&mut self, screen: Box<dyn Screen>) {
        self.current_screen = screen;
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new().expect("Failed to create app")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_creation() {
        let app = App::new();
        assert!(app.is_ok());
    }

    #[tokio::test]
    async fn test_app_state_transitions() {
        let mut app = App::new().unwrap();
        assert_eq!(app.state, AppState::Running);

        app.state = AppState::ShouldQuit;
        assert_eq!(app.state, AppState::ShouldQuit);
    }
}
