use crate::component_status::{is_component_installed, python_interpreters};
use crate::config::InstallerConfig;
use crate::hardware::{detect_hardware, run_preflight_checks};
use crate::installer::{run_installation, InstallerEvent};
use crate::state::{
    default_components, Category, Component, HardwareState, InstallStatus, PreflightResult, Stage,
};
use crate::widgets::benchmarks_page::{load_benchmark_results, render_benchmark_page};
use chrono::Local;
use crossterm::event::KeyModifiers;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, Gauge, List, ListItem, Paragraph, Row, Table, Wrap,
};
use ratatui::Frame;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WelcomePage {
    Introduction,
    UseCases,
    Walkthrough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Line,
    Raw,
}

impl InputMode {
    fn label(self) -> &'static str {
        match self {
            InputMode::Line => "line",
            InputMode::Raw => "raw",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskStatus {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

impl TaskStatus {
    fn icon(self) -> &'static str {
        match self {
            TaskStatus::Pending => "○",
            TaskStatus::Running => "…",
            TaskStatus::Done => "✓",
            TaskStatus::Failed => "✗",
            TaskStatus::Skipped => "⊘",
        }
    }

    fn label(self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::Running => "running",
            TaskStatus::Done => "done",
            TaskStatus::Failed => "failed",
            TaskStatus::Skipped => "skipped",
        }
    }

    fn color(self) -> Color {
        match self {
            TaskStatus::Pending => Color::Yellow,
            TaskStatus::Running => Color::Cyan,
            TaskStatus::Done => Color::Green,
            TaskStatus::Failed => Color::Red,
            TaskStatus::Skipped => Color::DarkGray,
        }
    }
}

#[derive(Debug)]
pub struct App {
    pub stage: Stage,
    pub components: Vec<Component>,
    pub selected_category: usize,
    pub selected_component: usize,
    pub logs: Vec<String>,
    pub errors: Vec<String>,
    pub hardware: HardwareState,
    pub preflight: PreflightResult,
    pub preflight_selection: usize,
    pub tick_count: u64,
    pub install_status: InstallStatus,
    pub config: InstallerConfig,
    pub config_selection: usize,
    pub config_dirty: bool,
    pub recovery_selection: usize,
    pub sudo_password: Option<String>,
    pub entering_password: bool,
    pub should_exit: bool,
    pub password_input: String,
    pub verification_reports: HashMap<String, Vec<String>>,
    pub welcome_page: WelcomePage,
    pub install_input_buffer: String,
    pub install_input_mode: InputMode,
    pub last_line_transient: bool,
    pub summary_scroll: u16,
    pub benchmark_tab_index: usize,
    install_input_sender: Option<Sender<String>>,
    install_receiver: Option<Receiver<InstallerEvent>>,
    hardware_receiver: Option<Receiver<anyhow::Result<HardwareState>>>,
}

impl App {
    pub fn new(scripts_dir: String) -> Self {
        let entering_password = unsafe { libc::geteuid() != 0 };
        let mut config = InstallerConfig::load_or_default(&scripts_dir).unwrap_or_else(|_| {
            // Fallback to a non-persisted config if directory is locked
            InstallerConfig::default_with_paths(
                &scripts_dir,
                "/tmp/mlstack/logs".into(),
                PathBuf::from("/tmp/mlstack/config.json"),
            )
        });
        config.scripts_dir = scripts_dir;
        Self {
            stage: Stage::Welcome,
            components: default_components(),
            selected_category: 0,
            selected_component: 0,
            logs: Vec::new(),
            errors: Vec::new(),
            hardware: HardwareState::default(),
            preflight: PreflightResult::default(),
            preflight_selection: 0,
            tick_count: 0,
            install_status: InstallStatus::default(),
            config,
            config_selection: 0,
            config_dirty: false,
            recovery_selection: 0,
            sudo_password: None,
            entering_password,
            should_exit: false,
            password_input: String::new(),
            verification_reports: HashMap::new(),
            welcome_page: WelcomePage::Introduction,
            install_input_buffer: String::new(),
            install_input_mode: InputMode::Line,
            last_line_transient: false,
            summary_scroll: 0,
            benchmark_tab_index: 0,
            install_input_sender: None,
            install_receiver: None,
            hardware_receiver: None,
        }
    }

    pub fn on_tick(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
        self.poll_hardware();
        self.poll_installer();
    }

    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        match self.stage {
            Stage::Welcome => match key.code {
                KeyCode::Right | KeyCode::Tab => {
                    self.welcome_page = match self.welcome_page {
                        WelcomePage::Introduction => WelcomePage::UseCases,
                        WelcomePage::UseCases => WelcomePage::Walkthrough,
                        WelcomePage::Walkthrough => WelcomePage::Introduction,
                    };
                }
                KeyCode::Left => {
                    self.welcome_page = match self.welcome_page {
                        WelcomePage::Introduction => WelcomePage::Walkthrough,
                        WelcomePage::UseCases => WelcomePage::Introduction,
                        WelcomePage::Walkthrough => WelcomePage::UseCases,
                    };
                }
                KeyCode::Enter => {
                    if self.entering_password {
                        if !self.password_input.is_empty() {
                            self.sudo_password = Some(self.password_input.clone());
                            self.entering_password = false;
                        } else {
                            self.errors.push("Password required for sudo".into());
                        }
                        return;
                    }
                    self.stage = Stage::HardwareDetect;
                    self.start_hardware_detection();
                }
                KeyCode::Char('q') | KeyCode::Char('Q') => {
                    self.should_exit = true;
                }
                KeyCode::Char(c) => {
                    if self.entering_password {
                        if c == '\n' || c == '\r' {
                            return;
                        }
                        self.password_input.push(c);
                    }
                }
                KeyCode::Backspace => {
                    if self.entering_password {
                        self.password_input.pop();
                    }
                }
                _ => {}
            },
            Stage::HardwareDetect => match key.code {
                KeyCode::Enter => {
                    self.preflight = run_preflight_checks(
                        &self.hardware.system,
                        &self.hardware.gpu,
                        self.sudo_password.as_deref(),
                    );
                    self.preflight_selection = 0;
                    self.stage = Stage::Preflight;
                }
                KeyCode::Esc => self.stage = Stage::Welcome,
                KeyCode::Char('q') => self.stage = Stage::Recovery,
                _ => {}
            },
            Stage::Preflight => match key.code {
                KeyCode::Up => self.move_preflight_selection(-1),
                KeyCode::Down => self.move_preflight_selection(1),
                KeyCode::Enter => {
                    if self.preflight.can_continue {
                        self.refresh_component_statuses();
                        self.stage = Stage::ComponentSelect;
                    } else {
                        self.errors
                            .push("Preflight checks failed; resolve critical issues".into());
                        self.stage = Stage::Recovery;
                    }
                }
                KeyCode::Esc => self.stage = Stage::HardwareDetect,
                KeyCode::Char('q') => self.stage = Stage::Recovery,
                _ => {}
            },
            Stage::ComponentSelect => match key.code {
                KeyCode::Up => self.move_selection(-1),
                KeyCode::Down => self.move_selection(1),
                KeyCode::Left => self.change_category(-1),
                KeyCode::Right => self.change_category(1),
                KeyCode::Char(' ') => self.toggle_component(),
                KeyCode::Char('a') => self.toggle_all(),
                KeyCode::Enter => self.stage = Stage::Configuration,
                KeyCode::Esc => self.stage = Stage::Preflight,
                KeyCode::Char('q') => self.stage = Stage::Recovery,
                _ => {}
            },
            Stage::Configuration => match key.code {
                KeyCode::Up => self.move_config_selection(-1),
                KeyCode::Down => self.move_config_selection(1),
                KeyCode::Enter => self.activate_config_selection(),
                KeyCode::Char('s') => self.save_config(),
                KeyCode::Char('n') => self.stage = Stage::Confirm,
                KeyCode::Esc => self.stage = Stage::ComponentSelect,
                KeyCode::Char('q') => self.stage = Stage::Recovery,
                _ => {}
            },
            Stage::Confirm => match key.code {
                KeyCode::Enter => self.start_installation(),
                KeyCode::Esc => self.stage = Stage::Configuration,
                KeyCode::Char('q') => self.stage = Stage::Recovery,
                _ => {}
            },
            Stage::Installing => match key.code {
                KeyCode::Char('q') => {
                    self.errors.push("Installation cancelled by user".into());
                    self.stage = Stage::Recovery;
                }
                KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.install_input_mode = match self.install_input_mode {
                        InputMode::Line => InputMode::Raw,
                        InputMode::Raw => InputMode::Line,
                    };
                }
                KeyCode::Backspace => {
                    self.install_input_buffer.pop();
                }
                KeyCode::Enter => {
                    self.flush_install_input();
                }
                KeyCode::Char(c) => {
                    if !key.modifiers.contains(KeyModifiers::CONTROL) {
                        self.install_input_buffer.push(c);
                        if self.install_input_mode == InputMode::Raw {
                            self.send_install_input(c.to_string());
                        }
                    }
                }
                _ => {}
            },
            Stage::Complete => match key.code {
                KeyCode::Esc | KeyCode::Char('q') => self.stage = Stage::Recovery,
                KeyCode::Char('b') | KeyCode::Char('B') => {
                    self.stage = Stage::Benchmarks;
                    self.benchmark_tab_index = 0;
                }
                KeyCode::Up => {
                    self.summary_scroll = self.summary_scroll.saturating_sub(1);
                }
                KeyCode::Down => {
                    self.summary_scroll = self.summary_scroll.saturating_add(1);
                }
                KeyCode::PageUp => {
                    self.summary_scroll = self.summary_scroll.saturating_sub(10);
                }
                KeyCode::PageDown => {
                    self.summary_scroll = self.summary_scroll.saturating_add(10);
                }
                _ => {}
            },
            Stage::Benchmarks => match key.code {
                KeyCode::Left => {
                    if self.benchmark_tab_index > 0 {
                        self.benchmark_tab_index -= 1;
                    }
                }
                KeyCode::Right => {
                    if self.benchmark_tab_index < 6 {
                        self.benchmark_tab_index += 1;
                    }
                }
                KeyCode::Esc | KeyCode::Char('q') => self.stage = Stage::Complete,
                _ => {}
            },
            Stage::Recovery => match key.code {
                KeyCode::Up => self.move_recovery_selection(-1),
                KeyCode::Down => self.move_recovery_selection(1),
                KeyCode::Enter => self.activate_recovery_selection(),
                KeyCode::Char('q') => self.should_exit = true,
                _ => {}
            },
        }
    }

    pub fn draw(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Length(5),
                    Constraint::Min(5),
                    Constraint::Length(3),
                ]
                .as_ref(),
            )
            .split(frame.size());

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
        ];
        let title = Paragraph::new(Text::from(title_lines))
            .block(Block::default().borders(Borders::ALL).title("Installer"));
        frame.render_widget(title, chunks[0]);

        match self.stage {
            Stage::Welcome => self.draw_welcome(frame, chunks[1]),
            Stage::HardwareDetect => self.draw_hardware(frame, chunks[1]),
            Stage::Preflight => self.draw_preflight(frame, chunks[1]),
            Stage::ComponentSelect => self.draw_component_select(frame, chunks[1]),
            Stage::Configuration => self.draw_configuration(frame, chunks[1]),
            Stage::Confirm => self.draw_confirm(frame, chunks[1]),
            Stage::Installing => self.draw_installing(frame, chunks[1]),
            Stage::Complete => self.draw_complete(frame, chunks[1]),
            Stage::Benchmarks => self.draw_benchmarks(frame, chunks[1]),
            Stage::Recovery => self.draw_recovery(frame, chunks[1]),
        }

        let footer = Paragraph::new(format!(
            "Stage: {:?} | Keys: {} | Logs: {} | {}",
            self.stage,
            self.stage_keymap(),
            self.logs.len(),
            Local::now().format("%H:%M:%S")
        ))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(footer, chunks[2]);
    }

    fn stage_keymap(&self) -> &'static str {
        match self.stage {
            Stage::Welcome => "Enter start • Q recovery",
            Stage::HardwareDetect => "Enter preflight • Esc back • Q recovery",
            Stage::Preflight => "↑/↓ select • Enter continue • Esc back • Q recovery",
            Stage::ComponentSelect => {
                "↑/↓ select • ←/→ category • Space toggle • A toggle all • Enter config • Esc back"
            }
            Stage::Configuration => "↑/↓ select • Enter toggle • S save • N next • Esc back",
            Stage::Confirm => "Enter install • Esc back • Q recovery",
            Stage::Installing => "Q recovery",
            Stage::Complete => "Esc recovery • B benchmarks",
            Stage::Benchmarks => "←/→ tabs • Esc/B back • Q quit",
            Stage::Recovery => "↑/↓ select • Enter apply • Q quit",
        }
    }

    fn spinner(&self) -> &'static str {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let idx = (self.tick_count % FRAMES.len() as u64) as usize;
        FRAMES[idx]
    }

    fn draw_welcome(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        use ratatui::widgets::canvas::{Canvas, Line as CanvasLine, Rectangle};

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)])
            .split(area);

        let content_area = chunks[0];
        let footer_area = chunks[1];

        match self.welcome_page {
            WelcomePage::Introduction => {
                let sub_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(content_area);

                // Left Side: ASCII Logo & Intro
                let logo = r#"
      __  __ _      ____  _             _    
     |  \/  | |    / ___|| |_ __ _  ___| | __
     | |\/| | |    \___ \| __/ _` |/ __| |/ /
     | |  | | |___  ___) | || (_| | (__|   < 
     |_|  |_|_____||____/ \__\__,_|\___|_|\_\
                "#;
                let intro_text = vec![
                    Line::from(vec![Span::styled(
                        logo,
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "Rusty-Stack ",
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw("is a high-performance"),
                    ]),
                    Line::from("ML infrastructure toolkit for AMD GPUs."),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("✓ ", Style::default().fg(Color::Green)),
                        Span::raw("Automated ROCm Toolchain Setup"),
                    ]),
                    Line::from(vec![
                        Span::styled("✓ ", Style::default().fg(Color::Green)),
                        Span::raw("Verified AI Extensions (DeepSpeed, vLLM)"),
                    ]),
                    Line::from(vec![
                        Span::styled("✓ ", Style::default().fg(Color::Green)),
                        Span::raw("Real-time Hardware Performance Benchmarks"),
                    ]),
                ];
                let intro_para = Paragraph::new(Text::from(intro_text))
                    .block(Block::default().borders(Borders::ALL).title("Overview"))
                    .wrap(Wrap { trim: true });
                frame.render_widget(intro_para, sub_chunks[0]);

                // Right Side: Visual Diagram
                let canvas = Canvas::default()
                    .block(Block::default().borders(Borders::ALL).title("Architecture"))
                    .paint(|ctx| {
                        // Main Hub
                        ctx.draw(&Rectangle {
                            x: 40.0,
                            y: 40.0,
                            width: 20.0,
                            height: 20.0,
                            color: Color::Cyan,
                        });
                        ctx.print(42.0, 50.0, "ROCm");

                        // Connection lines
                        ctx.draw(&CanvasLine {
                            x1: 50.0,
                            y1: 60.0,
                            x2: 30.0,
                            y2: 80.0,
                            color: Color::White,
                        });
                        ctx.draw(&CanvasLine {
                            x1: 50.0,
                            y1: 60.0,
                            x2: 70.0,
                            y2: 80.0,
                            color: Color::White,
                        });
                        ctx.draw(&CanvasLine {
                            x1: 50.0,
                            y1: 40.0,
                            x2: 50.0,
                            y2: 20.0,
                            color: Color::White,
                        });

                        // Nodes
                        ctx.print(20.0, 85.0, "PyTorch");
                        ctx.print(65.0, 85.0, "Docker");
                        ctx.print(45.0, 15.0, "Hardware");
                    })
                    .x_bounds([0.0, 100.0])
                    .y_bounds([0.0, 100.0]);
                frame.render_widget(canvas, sub_chunks[1]);
            }
            WelcomePage::UseCases => {
                let sub_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(content_area);

                // Left: Textual Use Cases
                let cases = vec![
                    Line::from(""),
                    Line::from(vec![Span::styled(
                        "🚀 Large Language Models",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )]),
                    Line::from("   Optimize vLLM and DeepSpeed for fast inference."),
                    Line::from(""),
                    Line::from(vec![Span::styled(
                        "🧪 Deep Learning Research",
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    )]),
                    Line::from("   Stable PyTorch builds with full HIP support."),
                    Line::from(""),
                    Line::from(vec![Span::styled(
                        "📦 Enterprise Deployment",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    )]),
                    Line::from("   Reproducible Docker environments for production."),
                ];
                let cases_para = Paragraph::new(Text::from(cases))
                    .block(Block::default().borders(Borders::ALL).title("Core Focus"))
                    .wrap(Wrap { trim: true });
                frame.render_widget(cases_para, sub_chunks[0]);

                // Right: Component Stack Diagram
                let canvas = Canvas::default()
                    .block(Block::default().borders(Borders::ALL).title("Tech Stack"))
                    .paint(|ctx| {
                        // Layers
                        // Base: Hardware
                        ctx.draw(&Rectangle {
                            x: 10.0,
                            y: 10.0,
                            width: 80.0,
                            height: 10.0,
                            color: Color::DarkGray,
                        });
                        ctx.print(40.0, 14.0, "AMD GPU (GFX)");

                        // ROCm Layer
                        ctx.draw(&Rectangle {
                            x: 10.0,
                            y: 20.0,
                            width: 80.0,
                            height: 10.0,
                            color: Color::Blue,
                        });
                        ctx.print(42.0, 24.0, "ROCm / HIP");

                        // AI Frameworks
                        ctx.draw(&CanvasLine {
                            x1: 50.0,
                            y1: 30.0,
                            x2: 30.0,
                            y2: 40.0,
                            color: Color::White,
                        });
                        ctx.draw(&Rectangle {
                            x: 15.0,
                            y: 40.0,
                            width: 30.0,
                            height: 15.0,
                            color: Color::Cyan,
                        });
                        ctx.print(25.0, 47.5, "PyTorch");

                        // Inference
                        ctx.draw(&CanvasLine {
                            x1: 50.0,
                            y1: 30.0,
                            x2: 70.0,
                            y2: 40.0,
                            color: Color::White,
                        });
                        ctx.draw(&Rectangle {
                            x: 55.0,
                            y: 40.0,
                            width: 30.0,
                            height: 15.0,
                            color: Color::Green,
                        });
                        ctx.print(62.0, 47.5, "vLLM / DS");
                    })
                    .x_bounds([0.0, 100.0])
                    .y_bounds([0.0, 100.0]);
                frame.render_widget(canvas, sub_chunks[1]);
            }
            WelcomePage::Walkthrough => {
                let lines = vec![
                    Line::from(""),
                    Line::from(vec![Span::styled(
                        "📖 Installation Walkthrough",
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    )]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "1. Hardware Check ",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("✓", Style::default().fg(Color::Green)),
                        Span::raw(" - Detects GPU, VRAM, and ROCm compatibility"),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "2. Pre-flight     ",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("⚠", Style::default().fg(Color::Yellow)),
                        Span::raw(" - Verifies OS dependencies and environment"),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "3. Selection      ",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("☐", Style::default().fg(Color::Cyan)),
                        Span::raw(" - Choose components (Core, Extensions, Tools)"),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "4. Configuration  ",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("⚙", Style::default().fg(Color::Magenta)),
                        Span::raw(" - Set paths and version preferences"),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "5. Installation   ",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("⟳", Style::default().fg(Color::Cyan)),
                        Span::raw(" - Automated build and setup process"),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "6. Verification   ",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("?", Style::default().fg(Color::Blue)),
                        Span::raw(" - Run diagnostics to ensure success"),
                    ]),
                ];
                let text_widget = Paragraph::new(Text::from(lines))
                    .block(Block::default().borders(Borders::ALL).title("Workflow"))
                    .wrap(Wrap { trim: true });
                frame.render_widget(text_widget, content_area);
            }
        }

        // Render Footer
        let page_num = match self.welcome_page {
            WelcomePage::Introduction => "1/3",
            WelcomePage::UseCases => "2/3",
            WelcomePage::Walkthrough => "3/3",
        };

        let footer_text = vec![Line::from(vec![
            Span::styled("Page ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                page_num,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Tab/→", Style::default().fg(Color::Yellow)),
            Span::raw(" next  "),
            Span::styled("←", Style::default().fg(Color::Yellow)),
            Span::raw(" prev  "),
            Span::styled("Enter", Style::default().fg(Color::Green)),
            Span::raw(" start  "),
            Span::styled("Q", Style::default().fg(Color::Red)),
            Span::raw(" quit"),
        ])];

        let footer_paragraph =
            Paragraph::new(Text::from(footer_text)).block(Block::default().borders(Borders::TOP));
        frame.render_widget(footer_paragraph, footer_area);
    }

    fn draw_hardware(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let status_color = if self.hardware.progress < 1.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        // Create a more visually appealing layout with color coding and symbols
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(area);

        // Header with status
        let header = Paragraph::new(Text::from(vec![Line::from(vec![
            Span::styled(
                "🌐 Hardware Detection",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(self.spinner(), Style::default().fg(status_color)),
            Span::raw(" "),
            Span::styled(
                self.hardware.status.clone(),
                Style::default().fg(status_color),
            ),
        ])]))
        .block(Block::default().borders(Borders::ALL))
        .wrap(Wrap { trim: true });
        frame.render_widget(header, chunks[0]);

        // Hardware information
        let info_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(chunks[1]);

        // Left side: System info
        let system_lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("🖥️  System", Style::default().fg(Color::Blue)),
                Span::raw(" "),
                Span::styled(
                    self.hardware.system.distribution.clone(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("🧠  Kernel", Style::default().fg(Color::Blue)),
                Span::raw(" "),
                Span::styled(
                    self.hardware.system.kernel.clone(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("💻  CPU", Style::default().fg(Color::Blue)),
                Span::raw(" "),
                Span::styled(
                    self.hardware.system.cpu_model.clone(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("💾  Memory", Style::default().fg(Color::Blue)),
                Span::raw(" "),
                Span::styled(
                    format!("{:.1} GB", self.hardware.system.memory_gb),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("💾  Storage", Style::default().fg(Color::Blue)),
                Span::raw(" "),
                Span::styled(
                    format!(
                        "{:.1} GB (free {:.1} GB)",
                        self.hardware.system.storage_gb, self.hardware.system.storage_available_gb
                    ),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
        ];

        // Right side: GPU info
        let gpu_lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("🖥️  GPU", Style::default().fg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    self.hardware.gpu.model.clone(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("🏗️  Architecture", Style::default().fg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    self.hardware.gpu.architecture.clone(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("🔧  ROCm", Style::default().fg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    if self.hardware.gpu.rocm_version.is_empty() {
                        "unknown".to_string()
                    } else {
                        self.hardware.gpu.rocm_version.clone()
                    },
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("💰  GPU Count", Style::default().fg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    self.hardware.gpu.gpu_count.to_string(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("💾  GPU Memory", Style::default().fg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    format!("{:.1} GB", self.hardware.gpu.memory_gb),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("🌡️  GPU Temp", Style::default().fg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    self.hardware
                        .gpu
                        .temperature_c
                        .map(|v| format!("{:.1} C", v))
                        .unwrap_or_else(|| "n/a".into()),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("⚡  GPU Power", Style::default().fg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    self.hardware
                        .gpu
                        .power_watts
                        .map(|v| format!("{:.1} W", v))
                        .unwrap_or_else(|| "n/a".into()),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
        ];

        let system_panel = Paragraph::new(Text::from(system_lines))
            .block(Block::default().borders(Borders::ALL).title("System"))
            .wrap(Wrap { trim: true });
        frame.render_widget(system_panel, info_chunks[0]);

        let gpu_panel = Paragraph::new(Text::from(gpu_lines))
            .block(Block::default().borders(Borders::ALL).title("GPU"))
            .wrap(Wrap { trim: true });
        frame.render_widget(gpu_panel, info_chunks[1]);

        // Footer with instructions
        let footer_lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "← Press Enter to run preflight checks",
                Style::default().fg(Color::Yellow),
            )),
            Line::from(Span::styled(
                "← Press Q to quit",
                Style::default().fg(Color::Red),
            )),
        ];
        let footer = Paragraph::new(Text::from(footer_lines))
            .block(Block::default().borders(Borders::ALL))
            .wrap(Wrap { trim: true });
        frame.render_widget(footer, chunks[2]);
    }

    fn draw_preflight(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(68), Constraint::Percentage(32)].as_ref())
            .split(area);

        let header = Row::new(vec![
            Cell::from(Span::styled(
                "Check",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Status",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Message",
                Style::default().add_modifier(Modifier::BOLD),
            )),
        ])
        .style(Style::default().fg(Color::Magenta));

        let rows: Vec<Row> = self
            .preflight
            .checks
            .iter()
            .enumerate()
            .map(|(idx, check)| {
                let status_color = match check.status {
                    crate::state::PreflightStatus::Passed => Color::Green,
                    crate::state::PreflightStatus::Warning => Color::Yellow,
                    crate::state::PreflightStatus::Failed => Color::Red,
                };
                let mut row_style = Style::default();
                if idx == self.preflight_selection {
                    row_style = row_style.bg(Color::DarkGray).add_modifier(Modifier::BOLD);
                }
                Row::new(vec![
                    Cell::from(check.name.clone()),
                    Cell::from(Span::styled(
                        check.status.label(),
                        Style::default()
                            .fg(status_color)
                            .add_modifier(Modifier::BOLD),
                    )),
                    Cell::from(check.message.clone()),
                ])
                .style(row_style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Percentage(38),
                Constraint::Length(10),
                Constraint::Percentage(52),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Preflight Checks"),
        )
        .column_spacing(1);
        frame.render_widget(table, chunks[0]);

        let selected = self.preflight.checks.get(self.preflight_selection);
        let summary_color = if self.preflight.can_continue {
            Color::Green
        } else {
            Color::Red
        };
        let mut summary_lines = vec![
            Line::from(Span::styled(
                self.preflight.summary.clone(),
                Style::default()
                    .fg(summary_color)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(format!(
                "Checks: {}  |  Score: {}",
                self.preflight.checks.len(),
                self.preflight.total_score
            )),
            Line::from(format!(
                "Passed: {}  Warnings: {}  Failed: {}",
                self.preflight.passed_count,
                self.preflight.warning_count,
                self.preflight.failed_count
            )),
            Line::from(format!(
                "Can continue: {}",
                if self.preflight.can_continue {
                    "yes"
                } else {
                    "no"
                }
            )),
            Line::from(""),
        ];

        if let Some(check) = selected {
            summary_lines.push(Line::from(Span::styled(
                format!("Selected: {}", check.name),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            summary_lines.push(Line::from(format!("Message: {}", check.message)));
            summary_lines.push(Line::from(format!("Details: {}", check.details)));
        }

        summary_lines.push(Line::from(""));
        summary_lines.push(Line::from("↑/↓ select  •  Enter continue  •  Q recovery"));

        let summary = Paragraph::new(Text::from(summary_lines))
            .block(Block::default().borders(Borders::ALL).title("Summary"))
            .wrap(Wrap { trim: true });
        frame.render_widget(summary, chunks[1]);
    }

    fn draw_component_select(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(45),
                Constraint::Percentage(30),
            ])
            .split(area);

        let categories = [
            (Category::Environment, "🌍", "Environment"),
            (Category::Foundation, "🔧", "Foundation"),
            (Category::Core, "⚙️", "Core"),
            (Category::Extension, "📦", "Extensions"),
            (Category::Verification, "✅", "Verification"),
            (Category::Performance, "📊", "Performance"),
        ];
        let category_items: Vec<ListItem> = categories
            .iter()
            .enumerate()
            .map(|(idx, (_, icon, label))| {
                let style = if idx == self.selected_category {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(Span::styled(
                    format!("{} {}", icon, label),
                    style,
                )))
            })
            .collect();

        let category_list = List::new(category_items)
            .block(Block::default().borders(Borders::ALL).title("Categories"));
        frame.render_widget(category_list, chunks[0]);

        let current_category = categories[self.selected_category].0;
        let filtered: Vec<(usize, &Component)> = self
            .components
            .iter()
            .enumerate()
            .filter(|(_, c)| c.category == current_category)
            .collect();

        let component_items: Vec<ListItem> = filtered
            .iter()
            .enumerate()
            .map(|(idx, (_, comp))| {
                let selected = idx == self.selected_component;

                // Category-specific icon
                let icon = match comp.category {
                    Category::Environment => "🌍",
                    Category::Foundation => "🔧",
                    Category::Core => "⚙️",
                    Category::Extension => "📦",
                    Category::Verification => "✅",
                    Category::Performance => "📊",
                };

                let indicator = if comp.selected { "☑" } else { "☐" };
                let status_indicator = if comp.installed { "✓" } else { "○" };

                let line = format!(
                    "{} {} {} {} [{}]",
                    indicator, icon, comp.name, status_indicator, comp.estimate
                );

                let style = if selected {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    if comp.installed {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default()
                    }
                };
                ListItem::new(Line::from(Span::styled(line, style)))
            })
            .collect();

        let component_list = List::new(component_items)
            .block(Block::default().borders(Borders::ALL).title("Components"));
        frame.render_widget(component_list, chunks[1]);

        let mut detail_lines = vec![Line::from(Span::styled(
            "Component Details",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))];
        if let Some((_, comp)) = filtered.get(self.selected_component) {
            detail_lines.push(Line::from(""));
            detail_lines.push(Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Gray)),
                Span::styled(comp.name.clone(), Style::default().fg(Color::White)),
            ]));
            detail_lines.push(Line::from(vec![
                Span::styled("Required: ", Style::default().fg(Color::Gray)),
                Span::styled(on_off(comp.required), Style::default().fg(Color::Yellow)),
            ]));
            detail_lines.push(Line::from(vec![
                Span::styled("Selected: ", Style::default().fg(Color::Gray)),
                Span::styled(on_off(comp.selected), Style::default().fg(Color::Cyan)),
            ]));

            // Category-specific status display
            let (status_label, status_color, status_icon) = match comp.category {
                Category::Environment => {
                    if comp.installed {
                        ("Configured", Color::Green, "✓")
                    } else {
                        ("Not configured", Color::Yellow, "○")
                    }
                }
                Category::Core => {
                    // Core components show their status based on installation
                    if comp.installed {
                        ("Installed", Color::Green, "✓")
                    } else {
                        ("Not installed", Color::Yellow, "○")
                    }
                }
                Category::Verification => {
                    if comp.installed {
                        ("Verified", Color::Green, "✓")
                    } else {
                        ("Unverified", Color::Yellow, "○")
                    }
                }
                Category::Performance => {
                    if comp.installed {
                        ("Benchmarked", Color::Green, "✓")
                    } else {
                        ("Pending benchmark", Color::Yellow, "○")
                    }
                }
                _ => {
                    if comp.installed {
                        ("Installed", Color::Green, "✓")
                    } else {
                        ("Not installed", Color::Yellow, "○")
                    }
                }
            };

            let status_line = Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{} {}", status_icon, status_label),
                    Style::default()
                        .fg(status_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]);
            detail_lines.push(status_line);
            detail_lines.push(Line::from(vec![
                Span::styled("Estimate: ", Style::default().fg(Color::Gray)),
                Span::styled(comp.estimate.clone(), Style::default().fg(Color::Magenta)),
            ]));
            detail_lines.push(Line::from(""));
            detail_lines.push(Line::from(comp.description.clone()));

            // Show category-specific info
            if comp.category == Category::Environment {
                detail_lines.push(Line::from(""));
                detail_lines.push(Line::from(Span::styled(
                    "Environment Configuration",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )));
                detail_lines.push(Line::from("Sets up persistent ROCm environment variables"));
                detail_lines.push(Line::from("for Python 3.12 across sessions."));
            } else if comp.category == Category::Verification {
                detail_lines.push(Line::from(""));
                detail_lines.push(Line::from(Span::styled(
                    "Verification Summary",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )));
                if let Some(report) = self.verification_reports.get(&comp.id) {
                    for line in report {
                        detail_lines.push(Line::from(line.clone()));
                    }
                } else {
                    detail_lines.push(Line::from("No verification report yet."));
                }
            } else if comp.category == Category::Performance {
                detail_lines.push(Line::from(""));
                detail_lines.push(Line::from(Span::styled(
                    "Performance Benchmarks",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )));
                detail_lines.push(Line::from("Run performance tests and benchmarks"));
                detail_lines.push(Line::from("to measure GPU throughput and efficiency."));
            }
        }
        detail_lines.push(Line::from(""));
        let selected_count = self.components.iter().filter(|c| c.selected).count();
        detail_lines.push(Line::from(format!(
            "Selected: {} of {}",
            selected_count,
            self.components.len()
        )));
        detail_lines.push(Line::from(""));
        detail_lines.push(Line::from(
            "Controls: ↑/↓ select • ←/→ category • Space toggle • Enter config • Q recovery",
        ));

        let detail_panel = Paragraph::new(Text::from(detail_lines))
            .block(Block::default().borders(Borders::ALL).title("Details"))
            .wrap(Wrap { trim: true });
        frame.render_widget(detail_panel, chunks[2]);
    }

    fn draw_configuration(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let items = self.config_items();
        let list_items: Vec<ListItem> = items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let style = if idx == self.config_selection {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(Span::styled(item.clone(), style)))
            })
            .collect();

        let hint = "Enter: toggle/cycle • s: save • n: next • esc: back";
        let mut lines = vec![Line::from("Configuration"), Line::from("")];
        lines.push(Line::from(format!(
            "Scripts Dir: {}",
            self.config.scripts_dir
        )));
        lines.push(Line::from(format!("Log Dir: {}", self.config.log_dir)));
        lines.push(Line::from(format!(
            "ROCm Install Path: {}",
            self.config.install_path
        )));
        lines.push(Line::from(""));
        let dirty_style = if self.config_dirty {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Green)
        };
        lines.push(Line::from(Span::styled(
            format!("Dirty: {}", if self.config_dirty { "yes" } else { "no" }),
            dirty_style,
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Selected setting:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        for line in self.config_help_lines() {
            lines.push(line);
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            hint,
            Style::default().fg(Color::Gray),
        )));

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)].as_ref())
            .split(area);

        let list =
            List::new(list_items).block(Block::default().borders(Borders::ALL).title("Settings"));
        frame.render_widget(list, chunks[0]);

        let detail = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Details"))
            .wrap(Wrap { trim: true });
        frame.render_widget(detail, chunks[1]);
    }

    fn draw_confirm(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let selected = self.selected_components();
        let mut lines = vec![
            Line::from(Span::styled(
                "Review your selection",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Selected Components:"),
            Line::from(""),
        ];
        for comp in selected {
            lines.push(Line::from(format!("• {} ({})", comp.name, comp.estimate)));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(format!(
            "ROCm Install Path: {}",
            self.config.install_path
        )));
        lines.push(Line::from(format!(
            "Batch Mode: {}",
            on_off(self.config.batch_mode)
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Press Enter to start installation",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )));
        let paragraph = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Confirm"))
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, area);
    }

    fn draw_installing(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        if area.height < 10 || area.width < 30 {
            frame.render_widget(
                Paragraph::new("Terminal too small to display installation UI"),
                area,
            );
            return;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5)].as_ref())
            .split(area);

        let clean_msg = self.install_status.message.trim_start_matches(|c: char| {
            c == '['
                || c == ']'
                || c == '⠋'
                || c == '⠙'
                || c == '⠹'
                || c == '⠸'
                || c == '⠼'
                || c == '⠴'
                || c == '⠦'
                || c == '⠧'
                || c == '⠇'
                || c == '⠏'
                || c.is_whitespace()
        });

        let percent = (self.install_status.progress * 100.0).round() as i32;
        let label = Span::styled(
            format!("{} {}% {}", self.spinner(), percent, clean_msg),
            Style::default()
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        );

        let gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Overall Progress"),
            )
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(self.install_status.progress as f64)
            .label(label);
        frame.render_widget(gauge, chunks[0]);

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
            .split(chunks[1]);

        let log_height = body[0].height.saturating_sub(2) as usize;
        let log_items: Vec<ListItem> = if self.logs.is_empty() {
            vec![ListItem::new(Line::from("Waiting for installer output..."))]
        } else {
            let start = self.logs.len().saturating_sub(log_height);
            self.logs[start..]
                .iter()
                .map(|line| ListItem::new(Line::from(line.clone())))
                .collect()
        };
        let log_list = List::new(log_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Installation Log"),
        );
        frame.render_widget(log_list, body[0]);

        let selected: Vec<&Component> = self.components.iter().filter(|c| c.selected).collect();
        let installed_count = selected.iter().filter(|c| c.installed).count();
        let failed_count = selected
            .iter()
            .filter(|c| !c.installed && c.progress > 0.0)
            .count();
        let pending_count = selected.iter().filter(|c| c.progress == 0.0).count();

        let base_lines = vec![
            Line::from(Span::styled(
                "Installation Status",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(format!("Installed: {}", installed_count)),
            Line::from(format!("Failed: {}", failed_count)),
            Line::from(format!("Pending: {}", pending_count)),
            Line::from(""),
            Line::from(format!("Current: {}", self.install_status.message)),
        ];

        let tail_lines = vec![
            Line::from(""),
            Line::from(format!(
                "Input mode: {} (Ctrl+R to toggle)",
                self.install_input_mode.label()
            )),
            Line::from(format!("Input buffer: {}", self.install_input_buffer)),
            Line::from("Enter sends line; raw mode sends keystrokes"),
            Line::from(""),
            Line::from(Span::styled(
                "Press Q for recovery",
                Style::default().fg(Color::Yellow),
            )),
        ];

        let available_lines = body[1].height.saturating_sub(2) as usize;
        let checklist_budget =
            available_lines.saturating_sub(base_lines.len() + tail_lines.len() + 1);
        let checklist_lines = self.checklist_lines(checklist_budget);

        let mut status_lines = base_lines;
        if !checklist_lines.is_empty() {
            status_lines.push(Line::from(""));
            status_lines.extend(checklist_lines);
        }
        status_lines.extend(tail_lines);

        let status_panel = Paragraph::new(Text::from(status_lines))
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: true });
        frame.render_widget(status_panel, body[1]);
    }

    fn draw_complete(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        frame.render_widget(Clear, area);
        let (installed, failed, skipped) = self.partition_components();
        let (benchmarks, tests) = self.count_log_keywords();
        let mut lines = vec![
            Line::from(Span::styled(
                "Installation Summary",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        lines.push(Line::from(Span::styled(
            "Installed:",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )));
        if installed.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for comp in installed.iter() {
                let (status, color) = if comp.installed {
                    ("installed", Color::Green)
                } else if comp.progress > 0.0 {
                    ("failed", Color::Red)
                } else {
                    ("skipped", Color::Yellow)
                };
                lines.push(Line::from(vec![
                    Span::raw("✓ "),
                    Span::raw(comp.name.clone()),
                    Span::raw(" ("),
                    Span::styled(status, Style::default().fg(color)),
                    Span::raw(")"),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Failed:",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
        if failed.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for comp in failed.iter() {
                lines.push(Line::from(vec![
                    Span::raw("✗ "),
                    Span::styled(comp.name.clone(), Style::default().fg(Color::Red)),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Skipped:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )));
        if skipped.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for comp in skipped.iter() {
                lines.push(Line::from(vec![
                    Span::raw("• "),
                    Span::styled(comp.name.clone(), Style::default().fg(Color::DarkGray)),
                ]));
            }
        }

        let (env, verification) = self.partition_categories();
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Environment:",
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )));
        if env.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for comp in env.iter() {
                let (status_text, color) = if comp.installed {
                    ("configured", Color::Green)
                } else {
                    ("pending", Color::Yellow)
                };
                lines.push(Line::from(vec![
                    Span::raw("• "),
                    Span::raw(comp.name.clone()),
                    Span::raw(" ("),
                    Span::styled(status_text, Style::default().fg(color)),
                    Span::raw(")"),
                ]));
            }
        }

        let env_details = self.env_summary();
        if !env_details.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Environment Details:",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            )));
            for (key, value) in env_details {
                lines.push(Line::from(vec![
                    Span::styled(format!("  {}: ", key), Style::default().fg(Color::Gray)),
                    Span::styled(value, Style::default().fg(Color::Cyan)),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Verification:",
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )));
        if verification.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for comp in verification.iter() {
                let (status, color) = match self.verification_reports.get(&comp.id) {
                    Some(_) => {
                        let s = self.verification_status(&comp.id);
                        let c = match s {
                            "verified" => Color::Green,
                            "failed" => Color::Red,
                            "missing" => Color::Yellow,
                            _ => Color::Gray,
                        };
                        (s, c)
                    }
                    None if comp.installed => ("completed", Color::Green),
                    None => ("pending", Color::Yellow),
                };
                lines.push(Line::from(vec![
                    Span::raw("• "),
                    Span::raw(comp.name.clone()),
                    Span::raw(" ("),
                    Span::styled(status, Style::default().fg(color)),
                    Span::raw(")"),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Component Verification:",
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )));
        let selected = self.selected_components();
        if selected.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for comp in selected {
                let status = self.verification_status(&comp.id);
                let color = match status {
                    "verified" => Color::Green,
                    "failed" => Color::Red,
                    "missing" => Color::Yellow,
                    _ => Color::Gray,
                };
                lines.push(Line::from(vec![
                    Span::raw("• "),
                    Span::raw(comp.name.clone()),
                    Span::raw(" ("),
                    Span::styled(status, Style::default().fg(color)),
                    Span::raw(")"),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Verification Report:",
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )));
        let report_lines = self.verification_report_lines();
        if report_lines.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            lines.extend(report_lines);
        }

        lines.push(Line::from(""));
        let benchmarks_line = Line::from(vec![
            Span::styled("Benchmarks logged: ", Style::default().fg(Color::Gray)),
            Span::styled(benchmarks.to_string(), Style::default().fg(Color::Cyan)),
            Span::styled(" | ", Style::default().fg(Color::DarkGray)),
            Span::styled("Tests logged: ", Style::default().fg(Color::Gray)),
            Span::styled(tests.to_string(), Style::default().fg(Color::Cyan)),
        ]);
        lines.push(benchmarks_line);

        let total_lines = lines.len() as u16;
        let visible_height = area.height.saturating_sub(2);
        let max_scroll = total_lines.saturating_sub(visible_height);

        let paragraph = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title(format!(
                "Complete (↑↓ to scroll, Esc to recovery) [{}/{}]",
                self.summary_scroll, max_scroll
            )))
            .wrap(Wrap { trim: true })
            .scroll((self.summary_scroll, 0));
        frame.render_widget(paragraph, area);
    }

    fn draw_benchmarks(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let results = load_benchmark_results();
        render_benchmark_page(frame, area, &results, self.benchmark_tab_index);
    }

    fn draw_recovery(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)].as_ref())
            .split(area);

        let items = self.recovery_items();
        let list_items: Vec<ListItem> = items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let style = if idx == self.recovery_selection {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(Span::styled(item.clone(), style)))
            })
            .collect();

        let list = List::new(list_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Recovery Options"),
        );
        frame.render_widget(list, chunks[0]);

        let mut lines = vec![Line::from("Recovery / Diagnostics"), Line::from("")];
        if !self.errors.is_empty() {
            lines.push(Line::from(Span::styled(
                "Errors:",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )));
            for err in &self.errors {
                lines.push(Line::from(Span::styled(
                    format!("• {}", err),
                    Style::default().fg(Color::Red),
                )));
            }
        } else {
            lines.push(Line::from(Span::styled(
                "No errors captured.",
                Style::default().fg(Color::Green),
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from("Use ↑/↓ then Enter to select."));
        lines.push(Line::from("Press Q to quit immediately."));

        let paragraph = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, chunks[1]);
    }

    fn move_selection(&mut self, delta: i32) {
        let current_category = self.current_category();
        let filtered: Vec<&Component> = self
            .components
            .iter()
            .filter(|c| c.category == current_category)
            .collect();
        if filtered.is_empty() {
            self.selected_component = 0;
            return;
        }
        let len = filtered.len() as i32;
        let mut idx = self.selected_component as i32 + delta;
        if idx < 0 {
            idx = len - 1;
        }
        if idx >= len {
            idx = 0;
        }
        self.selected_component = idx as usize;
    }

    fn change_category(&mut self, delta: i32) {
        let categories_len = 6i32; // Environment, Foundation, Core, Extension, Verification, Performance
        let mut idx = self.selected_category as i32 + delta;
        if idx < 0 {
            idx = categories_len - 1;
        }
        if idx >= categories_len {
            idx = 0;
        }
        self.selected_category = idx as usize;
        self.selected_component = 0;
    }

    fn toggle_component(&mut self) {
        let current_category = self.current_category();
        let indices: Vec<usize> = self
            .components
            .iter()
            .enumerate()
            .filter(|(_, c)| c.category == current_category)
            .map(|(idx, _)| idx)
            .collect();
        if indices.is_empty() {
            return;
        }
        if self.selected_component >= indices.len() {
            self.selected_component = 0;
        }
        let index = indices[self.selected_component];
        if let Some(component) = self.components.get_mut(index) {
            component.selected = !component.selected;
        }
    }

    fn toggle_all(&mut self) {
        let any_selected = self.components.iter().any(|c| c.selected);
        for comp in self.components.iter_mut() {
            comp.selected = !any_selected;
        }
    }

    fn config_items(&self) -> Vec<String> {
        vec![
            format!("ROCm Install Path: {}", self.config.install_path),
            format!("Batch Mode: {}", on_off(self.config.batch_mode)),
            format!("Auto Confirm: {}", on_off(self.config.auto_confirm)),
            format!("Star ML Stack Repo: {}", on_off(self.config.star_repos)),
            format!(
                "Force Reinstall All: {}",
                on_off(self.config.force_reinstall)
            ),
            format!("Theme: {}", self.config.theme),
            format!("Performance Profile: {}", self.config.performance_profile),
            "Save Configuration".into(),
        ]
    }

    fn config_help_lines(&self) -> Vec<Line<'_>> {
        match self.config_selection {
            0 => vec![
                Line::from("ROCm install path used by installers."),
                Line::from("Default: /opt/rocm (system-wide ROCm)."),
            ],
            1 => vec![
                Line::from("Batch mode runs scripts non-interactively."),
                Line::from("Defaults are chosen when prompts appear."),
            ],
            2 => vec![
                Line::from("Auto confirm answers yes to prompts"),
                Line::from("when supported by the script."),
            ],
            3 => vec![
                Line::from("Star the ML Stack repository on GitHub."),
                Line::from("https://github.com/scooter-lacroix/Stan-s-ML-Stack"),
            ],
            4 => vec![
                Line::from("FORCE REINSTALL ALL COMPONENTS."),
                Line::from("Forces purging and re-downloading of everything."),
            ],
            5 => vec![
                Line::from("Theme affects TUI color styling."),
                Line::from("Switches between dark/light palettes."),
            ],
            6 => vec![
                Line::from("Performance profile adjusts installer tuning."),
                Line::from("Balanced/performance/efficiency presets."),
            ],
            7 => vec![
                Line::from("Persist current settings to config.json."),
                Line::from("Dirty=yes means there are unsaved changes."),
            ],
            _ => vec![Line::from("Select a setting for details.")],
        }
    }

    fn move_config_selection(&mut self, delta: i32) {
        let len = self.config_items().len() as i32;
        let mut idx = self.config_selection as i32 + delta;
        if idx < 0 {
            idx = len - 1;
        }
        if idx >= len {
            idx = 0;
        }
        self.config_selection = idx as usize;
    }

    fn move_preflight_selection(&mut self, delta: i32) {
        let len = self.preflight.checks.len() as i32;
        if len == 0 {
            self.preflight_selection = 0;
            return;
        }
        let mut idx = self.preflight_selection as i32 + delta;
        if idx < 0 {
            idx = len - 1;
        }
        if idx >= len {
            idx = 0;
        }
        self.preflight_selection = idx as usize;
    }

    fn activate_config_selection(&mut self) {
        match self.config_selection {
            1 => {
                self.config.batch_mode = !self.config.batch_mode;
                self.config_dirty = true;
            }
            2 => {
                self.config.auto_confirm = !self.config.auto_confirm;
                self.config_dirty = true;
            }
            3 => {
                self.config.star_repos = !self.config.star_repos;
                self.config_dirty = true;
            }
            4 => {
                self.config.force_reinstall = !self.config.force_reinstall;
                self.config_dirty = true;
            }
            5 => {
                self.config.theme = if self.config.theme == "dark" {
                    "light".into()
                } else {
                    "dark".into()
                };
                self.config_dirty = true;
            }
            6 => {
                let profiles = ["balanced", "performance", "efficiency"];
                let current = profiles
                    .iter()
                    .position(|p| *p == self.config.performance_profile)
                    .unwrap_or(0);
                let next = (current + 1) % profiles.len();
                self.config.performance_profile = profiles[next].into();
                self.config_dirty = true;
            }
            7 => self.save_config(),
            _ => {}
        }
    }

    fn save_config(&mut self) {
        let existing = std::fs::read_to_string(&self.config.config_path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok());
        match self.config.save(existing) {
            Ok(_) => {
                self.push_log("Configuration saved".into());
                self.config_dirty = false;
            }
            Err(err) => {
                self.errors.push(format!("Config save failed: {err}"));
            }
        }
    }

    fn recovery_items(&self) -> Vec<String> {
        vec![
            "Re-run hardware detection".into(),
            "Re-run preflight checks".into(),
            "Return to component selection".into(),
            "Exit installer".into(),
        ]
    }

    fn move_recovery_selection(&mut self, delta: i32) {
        let len = self.recovery_items().len() as i32;
        let mut idx = self.recovery_selection as i32 + delta;
        if idx < 0 {
            idx = len - 1;
        }
        if idx >= len {
            idx = 0;
        }
        self.recovery_selection = idx as usize;
    }

    fn activate_recovery_selection(&mut self) {
        match self.recovery_selection {
            0 => {
                self.stage = Stage::HardwareDetect;
                self.start_hardware_detection();
            }
            1 => {
                self.preflight = run_preflight_checks(
                    &self.hardware.system,
                    &self.hardware.gpu,
                    self.sudo_password.as_deref(),
                );
                self.preflight_selection = 0;
                self.stage = Stage::Preflight;
            }
            2 => {
                self.refresh_component_statuses();
                self.stage = Stage::ComponentSelect;
            }
            3 => self.should_exit = true,
            _ => {}
        }
    }

    fn start_installation(&mut self) {
        let selected = self.selected_components();
        if selected.is_empty() {
            self.logs.push("No components selected".into());
            return;
        }
        if unsafe { libc::geteuid() != 0 } && self.sudo_password.is_none() {
            self.errors
                .push("Sudo password required before installation".into());
            self.stage = Stage::Configuration;
            return;
        }

        // Set FORCE=true if force_reinstall is enabled
        if self.config.force_reinstall {
            std::env::set_var("FORCE", "true");
            std::env::set_var("PYTORCH_REINSTALL", "true");
        } else {
            std::env::remove_var("FORCE");
            std::env::remove_var("PYTORCH_REINSTALL");
        }

        self.stage = Stage::Installing;
        self.install_status.progress = 0.0;
        self.install_status.message = "Starting installation".into();
        let (tx, rx) = mpsc::channel();
        let (input_tx, input_rx) = mpsc::channel();
        let sudo_password = self.sudo_password.clone();
        let config = self.config.clone();
        self.install_input_sender = Some(input_tx);
        self.install_input_buffer.clear();
        thread::spawn(move || {
            run_installation(selected, config, sudo_password, tx, input_rx);
        });
        self.install_receiver = Some(rx);
    }

    fn poll_installer(&mut self) {
        let events: Vec<_> = if let Some(receiver) = &self.install_receiver {
            std::iter::from_fn(|| receiver.try_recv().ok()).collect()
        } else {
            vec![]
        };

        for event in events {
            match event {
                InstallerEvent::Log(line, is_transient) => self.push_log_ext(line, is_transient),
                InstallerEvent::Progress {
                    component_id,
                    progress,
                    message,
                } => {
                    if component_id != "__overall__" {
                        self.update_component_progress(&component_id, progress);
                    }
                    self.install_status.message = message.clone();
                    self.recalculate_overall_progress();
                    // Progress messages are usually transient
                    self.push_log_ext(message, true);
                }
                InstallerEvent::ComponentStart { component_id, name } => {
                    self.install_status.message = format!("Installing {}", name);
                    if let Some(comp) = self.components.iter_mut().find(|c| c.id == component_id) {
                        comp.progress = 0.05;
                        comp.installed = false;
                    }
                    self.recalculate_overall_progress();
                    self.push_log(self.install_status.message.clone());
                }
                InstallerEvent::ComponentComplete {
                    component_id,
                    success,
                    message,
                } => {
                    if let Some(comp) = self.components.iter_mut().find(|c| c.id == component_id) {
                        comp.installed = success;
                        comp.progress = 1.0;
                    }
                    self.recalculate_overall_progress();
                    if !success {
                        self.errors.push(message.clone());
                    }
                    self.push_log(message);
                }
                InstallerEvent::VerificationReport {
                    component_id,
                    lines,
                } => {
                    let cleaned = lines
                        .into_iter()
                        .map(|line| Self::sanitize_line(&line))
                        .collect();
                    self.verification_reports.insert(component_id, cleaned);
                }
                InstallerEvent::Finished { success } => {
                    self.install_status.completed = success;
                    self.install_input_sender = None;
                    self.install_input_buffer.clear();
                    self.recalculate_overall_progress();
                    self.install_status.progress = 1.0;
                    self.stage = Stage::Benchmarks;
                }
            }
        }
    }

    fn update_component_progress(&mut self, component_id: &str, progress: f32) {
        if let Some(component) = self.components.iter_mut().find(|c| c.id == component_id) {
            let next = progress.clamp(0.0, 1.0);
            if next > component.progress {
                component.progress = next;
            }
        }
    }

    fn recalculate_overall_progress(&mut self) {
        let selected: Vec<&Component> = self.components.iter().filter(|c| c.selected).collect();
        if selected.is_empty() {
            self.install_status.progress = 0.0;
            return;
        }
        let total: f32 = selected
            .iter()
            .map(|comp| comp.progress.clamp(0.0, 1.0))
            .sum();
        self.install_status.progress = (total / selected.len() as f32).clamp(0.0, 1.0);
    }

    fn install_task_status(&self, component: &Component) -> TaskStatus {
        if component.progress == 0.0 {
            TaskStatus::Pending
        } else if component.progress < 1.0 {
            TaskStatus::Running
        } else if component.installed {
            TaskStatus::Done
        } else {
            TaskStatus::Failed
        }
    }

    fn verification_task_status(&self, component_id: &str) -> TaskStatus {
        match self.verification_status(component_id) {
            "verified" => TaskStatus::Done,
            "failed" => TaskStatus::Failed,
            "missing" => TaskStatus::Failed,
            "unknown" => TaskStatus::Pending,
            "none" => TaskStatus::Pending,
            _ => TaskStatus::Pending,
        }
    }

    fn verify_task_status(&self, component: &Component) -> TaskStatus {
        if component.category == Category::Verification {
            return self.verification_task_status(&component.id);
        }
        if self.verification_reports.contains_key(&component.id) {
            return self.verification_task_status(&component.id);
        }
        if component.progress >= 1.0 && !component.installed {
            return TaskStatus::Skipped;
        }
        if component.progress >= 0.8 && component.progress < 1.0 {
            return TaskStatus::Running;
        }
        TaskStatus::Pending
    }

    fn checklist_lines(&self, max_lines: usize) -> Vec<Line<'_>> {
        if max_lines == 0 {
            return Vec::new();
        }
        let selected: Vec<&Component> = self.components.iter().filter(|c| c.selected).collect();
        let mut lines = Vec::new();
        let mut remaining = max_lines;

        lines.push(Line::from(Span::styled(
            "Checklist",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        remaining = remaining.saturating_sub(1);

        let mut truncated = false;
        for comp in selected {
            if remaining == 0 {
                truncated = true;
                break;
            }
            lines.push(Line::from(Span::styled(
                format!("• {}", comp.name),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )));
            remaining = remaining.saturating_sub(1);

            if remaining == 0 {
                truncated = true;
                break;
            }

            if comp.category != Category::Verification {
                lines.push(self.task_line("Install", self.install_task_status(comp)));
                remaining = remaining.saturating_sub(1);
            }

            if remaining == 0 {
                truncated = true;
                break;
            }

            lines.push(self.task_line("Verify", self.verify_task_status(comp)));
            remaining = remaining.saturating_sub(1);
        }

        if truncated {
            lines.push(Line::from("  …"));
        }

        lines
    }

    fn task_line(&self, label: &str, status: TaskStatus) -> Line<'_> {
        Line::from(vec![
            Span::styled(
                format!("  {}", status.icon()),
                Style::default().fg(status.color()),
            ),
            Span::raw(format!(" {} ({})", label, status.label())),
        ])
    }

    fn send_install_input(&self, payload: String) {
        if let Some(sender) = &self.install_input_sender {
            let _ = sender.send(payload);
        }
    }

    fn flush_install_input(&mut self) {
        if self.install_input_mode == InputMode::Raw {
            self.send_install_input("\n".to_string());
            return;
        }

        if self.install_input_buffer.is_empty() {
            self.send_install_input("\n".to_string());
            return;
        }

        let mut payload = self.install_input_buffer.clone();
        payload.push('\n');
        self.send_install_input(payload);
        self.install_input_buffer.clear();
    }

    fn start_hardware_detection(&mut self) {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let result = detect_hardware();
            let _ = tx.send(result);
        });
        self.hardware_receiver = Some(rx);
        self.hardware.status = "Detecting hardware...".into();
        self.hardware.progress = 0.1;
    }

    fn poll_hardware(&mut self) {
        let results: Vec<_> = if let Some(receiver) = &self.hardware_receiver {
            std::iter::from_fn(|| receiver.try_recv().ok()).collect()
        } else {
            vec![]
        };

        for result in results {
            match result {
                Ok(state) => {
                    self.hardware = state;
                }
                Err(err) => {
                    self.errors.push(err.to_string());
                }
            }
        }
    }

    fn refresh_component_statuses(&mut self) {
        let python_candidates = python_interpreters();
        for component in &mut self.components {
            if component.category == Category::Verification {
                continue;
            }
            component.installed = is_component_installed(component, &python_candidates);
            if component.installed {
                component.selected = false;
            }
        }
    }

    fn selected_components(&self) -> Vec<Component> {
        self.components
            .iter()
            .filter(|c| c.selected)
            .cloned()
            .collect()
    }

    fn current_category(&self) -> Category {
        match self.selected_category {
            0 => Category::Environment,
            1 => Category::Foundation,
            2 => Category::Core,
            3 => Category::Extension,
            4 => Category::Verification,
            _ => Category::Performance,
        }
    }

    fn sanitize_line(value: &str) -> String {
        let mut clean = String::with_capacity(value.len());
        let mut in_escape = false;
        for c in value.chars() {
            if c == '\x1b' {
                in_escape = true;
                continue;
            }
            if in_escape {
                if ('@'..='~').contains(&c) || c == 'm' {
                    in_escape = false;
                }
                continue;
            }
            if c == '\t' {
                clean.push(' ');
            } else if !c.is_control() {
                clean.push(c);
            }
        }
        clean
    }

    fn push_log(&mut self, line: String) {
        self.push_log_ext(line, false);
    }

    fn push_log_ext(&mut self, line: String, is_transient: bool) {
        let timestamp = Local::now().format("%H:%M:%S");
        let clean_line = Self::sanitize_line(&line);
        if clean_line.trim().is_empty() {
            return;
        }
        let entry = format!("[{}] {}", timestamp, clean_line);

        if is_transient && self.last_line_transient && !self.logs.is_empty() {
            // Replace the last line if it was also transient
            let last_idx = self.logs.len() - 1;
            self.logs[last_idx] = entry.clone();
        } else {
            self.logs.push(entry.clone());
        }

        self.last_line_transient = is_transient;

        if self.logs.len() > 2000 {
            self.logs.drain(0..500);
        }

        // Only write non-transient logs to disk to keep it clean
        if !is_transient {
            let log_dir = std::path::Path::new(&self.config.log_dir);
            let log_path = log_dir.join("rusty-stack.log");
            if std::fs::create_dir_all(log_dir).is_ok() {
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(log_path)
                    .and_then(|mut file| {
                        use std::io::Write;
                        writeln!(file, "{}", entry)
                    });
            }
        }
    }

    fn partition_categories(&self) -> (Vec<Component>, Vec<Component>) {
        let env = self
            .components
            .iter()
            .filter(|c| c.category == Category::Environment && c.selected)
            .cloned()
            .collect::<Vec<_>>();
        let verification = self
            .components
            .iter()
            .filter(|c| c.category == Category::Verification && c.selected)
            .cloned()
            .collect::<Vec<_>>();
        (env, verification)
    }

    fn env_summary(&self) -> Vec<(String, String)> {
        let mut entries = Vec::new();
        let env_path = std::path::Path::new(&self.config.install_path).join(".mlstack_env");
        let fallback =
            std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join(".mlstack_env");
        let path = if env_path.exists() {
            env_path
        } else {
            fallback
        };

        if let Ok(contents) = std::fs::read_to_string(&path) {
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("export ") {
                    let mut parts = rest.splitn(2, '=');
                    if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                        entries.push((
                            key.trim().to_string(),
                            value.trim().trim_matches('"').to_string(),
                        ));
                    }
                }
            }
        }

        let keys = [
            "ROCM_VERSION",
            "ROCM_CHANNEL",
            "GPU_ARCH",
            "ROCM_HOME",
            "ROCM_PATH",
            "HIP_PATH",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_ROCM_DEVICE",
        ];

        let mut filtered = Vec::new();
        for key in keys {
            if let Some((_, value)) = entries.iter().find(|(k, _)| k == key) {
                filtered.push((key.to_string(), Self::sanitize_line(value)));
            } else if let Ok(value) = std::env::var(key) {
                filtered.push((key.to_string(), Self::sanitize_line(&value)));
            }
        }
        filtered
    }

    fn verification_report_lines(&self) -> Vec<Line<'_>> {
        let mut lines = Vec::new();
        for comp in self.components.iter().filter(|c| c.selected) {
            if let Some(report) = self.verification_reports.get(&comp.id) {
                lines.push(Line::from(Span::styled(
                    format!("{} Report:", comp.name),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )));
                for entry in report {
                    if entry.trim().is_empty() {
                        lines.push(Line::from(""));
                    } else {
                        let style = if entry.contains("Verified") {
                            Style::default().fg(Color::Green)
                        } else if entry.contains("Failed") {
                            Style::default().fg(Color::Red)
                        } else if entry.contains("Missing") || entry.contains("Skipped") {
                            Style::default().fg(Color::Yellow)
                        } else {
                            Style::default()
                        };
                        lines.push(Line::from(Span::styled(format!("  {}", entry), style)));
                    }
                }
                lines.push(Line::from(""));
            }
        }
        lines
    }

    fn verification_status(&self, component_id: &str) -> &'static str {
        let Some(report) = self.verification_reports.get(component_id) else {
            return "none";
        };
        let mut has_verified = false;
        for line in report {
            if line.contains("Failed") {
                return "failed";
            }
            if line.contains("Missing") {
                return "missing";
            }
            if line.contains("Verified") {
                has_verified = true;
            }
        }
        if has_verified {
            "verified"
        } else {
            "unknown"
        }
    }

    fn count_log_keywords(&self) -> (usize, usize) {
        let mut benchmarks = 0usize;
        let mut tests = 0usize;
        for entry in &self.logs {
            let lower = entry.to_lowercase();
            if lower.contains("benchmark") {
                benchmarks += 1;
            }
            if lower.contains("test") || lower.contains("verify") {
                tests += 1;
            }
        }
        for report in self.verification_reports.values() {
            for entry in report {
                let lower = entry.to_lowercase();
                if lower.contains("benchmark") {
                    benchmarks += 1;
                }
                if lower.contains("test") || lower.contains("verify") {
                    tests += 1;
                }
            }
        }
        (benchmarks, tests)
    }

    fn partition_components(&self) -> (Vec<Component>, Vec<Component>, Vec<Component>) {
        let mut installed = Vec::new();
        let mut failed = Vec::new();
        let mut skipped = Vec::new();
        for comp in self.components.iter().filter(|c| c.selected) {
            if comp.installed {
                installed.push(comp.clone());
            } else if comp.progress > 0.0 {
                failed.push(comp.clone());
            } else {
                skipped.push(comp.clone());
            }
        }
        (installed, failed, skipped)
    }
}

fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}
