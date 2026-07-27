use crate::component_status::{is_component_installed, python_interpreters};
use crate::config::InstallerConfig;
use crate::hardware::{detect_hardware, run_preflight_checks};
use crate::installer::{run_installation, InstallerEvent};
use crate::installers::components::llama_cpp::{LlamaCppConfig, LlamaCppInstaller};
use crate::state::{
    default_components, Category, Component, HardwareState, InstallStatus, PreflightResult,
    RunMode, Stage,
};
use crate::telemetry::opt_in::{
    OptInGate, TELEMETRY_DESCRIPTION, TELEMETRY_DISABLE_LABEL, TELEMETRY_ENABLE_LABEL,
    TELEMETRY_PRIVACY_NOTE, TELEMETRY_STATUS_DISABLED, TELEMETRY_STATUS_ENABLED,
};
use crate::widgets::benchmarks_page::{
    export_benchmark_report_html, load_benchmark_results, render_benchmark_page,
};
use chrono::Local;
use crossterm::event::KeyModifiers;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
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

#[derive(Debug, Clone)]
struct UiNotice {
    message: String,
    color: Color,
    expires_at_tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComponentDetectionStatus {
    Pending,
    Running,
    Installed,
    NotInstalled,
    Skipped,
}

#[derive(Debug)]
enum ComponentStatusEvent {
    Started(usize),
    Finished { index: usize, installed: bool },
    Skipped(usize),
    Complete(Vec<Component>),
}

impl ComponentDetectionStatus {
    fn icon(self) -> &'static str {
        match self {
            ComponentDetectionStatus::Pending => "○",
            ComponentDetectionStatus::Running => "…",
            ComponentDetectionStatus::Installed => "✓",
            ComponentDetectionStatus::NotInstalled => "◇",
            ComponentDetectionStatus::Skipped => "⊘",
        }
    }

    fn label(self) -> &'static str {
        match self {
            ComponentDetectionStatus::Pending => "pending",
            ComponentDetectionStatus::Running => "checking",
            ComponentDetectionStatus::Installed => "installed",
            ComponentDetectionStatus::NotInstalled => "not installed",
            ComponentDetectionStatus::Skipped => "skipped",
        }
    }

    fn color(self) -> Color {
        match self {
            ComponentDetectionStatus::Pending => Color::Yellow,
            ComponentDetectionStatus::Running => Color::Cyan,
            ComponentDetectionStatus::Installed => Color::Green,
            ComponentDetectionStatus::NotInstalled => Color::Gray,
            ComponentDetectionStatus::Skipped => Color::DarkGray,
        }
    }
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
    pub confirm_selection: usize,
    pub config_dirty: bool,
    pub recovery_selection: usize,
    pub sudo_password: Option<String>,
    pub entering_password: bool,
    pub should_exit: bool,
    pub password_input: String,
    pub verification_reports: HashMap<String, Vec<String>>,
    pub install_input_buffer: String,
    pub install_input_mode: InputMode,
    pub last_line_transient: bool,
    pub summary_scroll: u16,
    pub benchmark_tab_index: usize,
    benchmark_notice: Option<UiNotice>,
    telemetry_gate: Option<OptInGate>,
    telemetry_prompt_pending: bool,
    telemetry_prompt_active: bool,
    install_log_popup: bool,
    install_log_scroll: usize,
    install_log_file_path: Option<String>,
    install_input_sender: Option<Sender<String>>,
    install_receiver: Option<Receiver<InstallerEvent>>,
    hardware_receiver: Option<Receiver<anyhow::Result<HardwareState>>>,
    component_status_receiver: Option<Receiver<ComponentStatusEvent>>,
    component_detection_statuses: Vec<ComponentDetectionStatus>,
    component_detection_done: bool,
    install_activity_tick: u64,
}

impl App {
    pub fn new(scripts_dir: String) -> Self {
        #[cfg(unix)]
        let entering_password = unsafe { libc::geteuid() != 0 };
        #[cfg(not(unix))]
        let entering_password = false;
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
            confirm_selection: 0,
            config_dirty: false,
            recovery_selection: 0,
            sudo_password: None,
            entering_password,
            should_exit: false,
            password_input: String::new(),
            verification_reports: HashMap::new(),
            install_input_buffer: String::new(),
            install_input_mode: InputMode::Line,
            last_line_transient: false,
            summary_scroll: 0,
            benchmark_tab_index: 0,
            benchmark_notice: None,
            telemetry_gate: OptInGate::new().ok(),
            telemetry_prompt_pending: false,
            telemetry_prompt_active: false,
            install_log_popup: false,
            install_log_scroll: 0,
            install_log_file_path: None,
            install_input_sender: None,
            install_receiver: None,
            hardware_receiver: None,
            component_status_receiver: None,
            component_detection_statuses: Vec::new(),
            component_detection_done: false,
            install_activity_tick: 0,
        }
    }

    pub fn on_tick(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
        if self
            .benchmark_notice
            .as_ref()
            .is_some_and(|notice| self.tick_count > notice.expires_at_tick)
        {
            self.benchmark_notice = None;
        }
        self.poll_hardware();
        self.poll_component_status_detection();
        self.poll_installer();
    }

    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        match self.stage {
            Stage::Welcome => match key.code {
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
                KeyCode::Char('q') => {
                    self.stage = Stage::Recovery;
                    self.errors.push("User quit".into());
                }
                KeyCode::Char(c) if self.entering_password => {
                    if c == '\n' || c == '\r' {
                        return;
                    }
                    self.password_input.push(c);
                }
                KeyCode::Char(_) => {}
                KeyCode::Backspace if self.entering_password => {
                    self.password_input.pop();
                }
                KeyCode::Backspace => {}
                KeyCode::Tab if self.entering_password => {
                    self.sudo_password = Some(self.password_input.clone());
                    self.entering_password = false;
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
                        self.start_component_status_detection();
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
            Stage::ComponentDetect => {
                if let KeyCode::Enter = key.code {
                    if self.component_detection_done {
                        self.stage = Stage::ComponentSelect;
                    }
                } else if let KeyCode::Char('q') = key.code {
                    self.stage = Stage::Recovery;
                }
            }
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
                KeyCode::Char('n') => {
                    self.confirm_selection = 0;
                    self.stage = Stage::Confirm;
                }
                KeyCode::Esc => self.stage = Stage::ComponentSelect,
                KeyCode::Char('q') => self.stage = Stage::Recovery,
                _ => {}
            },
            Stage::Confirm => match key.code {
                KeyCode::Up => self.move_confirm_selection(-1),
                KeyCode::Down => self.move_confirm_selection(1),
                KeyCode::Left => self.adjust_confirm_selection(-1),
                KeyCode::Right => self.adjust_confirm_selection(1),
                KeyCode::Enter => self.activate_confirm_selection(),
                KeyCode::Esc => self.stage = Stage::Configuration,
                KeyCode::Char('q') => self.stage = Stage::Recovery,
                _ => {}
            },
            Stage::Installing => match key.code {
                KeyCode::Char('l') | KeyCode::Char('L') => {
                    self.install_log_popup = !self.install_log_popup;
                }
                KeyCode::Esc if self.install_log_popup => {
                    self.install_log_popup = false;
                }
                KeyCode::Up if self.install_log_popup => {
                    self.install_log_scroll = self.install_log_scroll.saturating_sub(1);
                }
                KeyCode::Down if self.install_log_popup => {
                    self.install_log_scroll = self.install_log_scroll.saturating_add(1);
                }
                KeyCode::PageUp if self.install_log_popup => {
                    self.install_log_scroll = self.install_log_scroll.saturating_sub(20);
                }
                KeyCode::PageDown if self.install_log_popup => {
                    self.install_log_scroll = self.install_log_scroll.saturating_add(20);
                }
                KeyCode::Home if self.install_log_popup => {
                    self.install_log_scroll = 0;
                }
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
                KeyCode::Backspace if !self.install_log_popup => {
                    self.install_input_buffer.pop();
                }
                KeyCode::Backspace => {}
                KeyCode::Enter if !self.install_log_popup => {
                    if self.telemetry_prompt_active {
                        if let Some(gate) = self.telemetry_gate.as_ref() {
                            if gate.is_enabled() {
                                let gpu = self.hardware.gpu.clone();
                                let report = crate::installers::common::BuildReport::from_hardware(
                                    &gpu,
                                    std::env::consts::OS.to_string(),
                                    std::env::var("ID").unwrap_or_else(|_| "unknown".to_string()),
                                    std::time::Duration::from_secs(
                                        self.install_status.progress as u64,
                                    ),
                                    crate::installers::common::BuildReportArtifacts {
                                        git_commit: "unknown".into(),
                                        install_path: self.config.install_path.clone(),
                                        cmake_flags: vec![],
                                        verification_path: self.config.install_path.clone(),
                                        binary_version: "unknown".into(),
                                        was_prebuilt: false,
                                    },
                                );
                                crate::installers::common::submit_build_report(report);
                            }
                        }
                        self.telemetry_prompt_pending = false;
                        self.telemetry_prompt_active = false;
                        self.stage = Stage::Benchmarks;
                        return;
                    }
                    self.flush_install_input();
                }
                KeyCode::Enter => {}
                KeyCode::Char(c)
                    if !self.install_log_popup
                        && !key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    if self.telemetry_prompt_active {
                        match c {
                            'y' | 'Y' => {
                                if let Some(gate) = self.telemetry_gate.as_ref() {
                                    if gate.is_enabled() {
                                        let gpu = self.hardware.gpu.clone();
                                        let report =
                                            crate::installers::common::BuildReport::from_hardware(
                                                &gpu,
                                                std::env::consts::OS.to_string(),
                                                std::env::var("ID")
                                                    .unwrap_or_else(|_| "unknown".to_string()),
                                                std::time::Duration::from_secs(
                                                    self.install_status.progress as u64,
                                                ),
                                                crate::installers::common::BuildReportArtifacts {
                                                    git_commit: "unknown".into(),
                                                    install_path: self.config.install_path.clone(),
                                                    cmake_flags: vec![],
                                                    verification_path: self
                                                        .config
                                                        .install_path
                                                        .clone(),
                                                    binary_version: "unknown".into(),
                                                    was_prebuilt: false,
                                                },
                                            );
                                        crate::installers::common::submit_build_report(report);
                                    }
                                }
                                self.telemetry_prompt_pending = false;
                                self.telemetry_prompt_active = false;
                                self.stage = Stage::Benchmarks;
                            }
                            'n' | 'N' => {
                                self.telemetry_prompt_pending = false;
                                self.telemetry_prompt_active = false;
                                self.stage = Stage::Benchmarks;
                            }
                            _ => {
                                self.install_input_buffer.push(c);
                                if self.install_input_mode == InputMode::Raw {
                                    self.send_install_input(c.to_string());
                                }
                            }
                        }
                        return;
                    }
                    self.install_input_buffer.push(c);
                    if self.install_input_mode == InputMode::Raw {
                        self.send_install_input(c.to_string());
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
                KeyCode::Left if self.benchmark_tab_index > 0 => {
                    self.benchmark_tab_index -= 1;
                }
                KeyCode::Left => {}
                KeyCode::Right if self.benchmark_tab_index < 9 => {
                    self.benchmark_tab_index += 1;
                }
                KeyCode::Right => {}
                KeyCode::Char('e') | KeyCode::Char('E') => self.export_benchmark_report(),
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
            .split(frame.area());

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
            Stage::ComponentDetect => self.draw_component_detection(frame, chunks[1]),
            Stage::ComponentSelect => self.draw_component_select(frame, chunks[1]),
            Stage::Configuration => self.draw_configuration(frame, chunks[1]),
            Stage::Confirm => self.draw_confirm(frame, chunks[1]),
            Stage::Installing => self.draw_installing(frame, chunks[1]),
            Stage::Complete => self.draw_complete(frame, chunks[1]),
            Stage::Benchmarks => self.draw_benchmarks(frame, chunks[1]),
            Stage::Recovery => self.draw_recovery(frame, chunks[1]),
        }
        self.draw_notice(frame, chunks[1]);

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
            Stage::ComponentDetect => "checking components • Q recovery",
            Stage::ComponentSelect => {
                "↑/↓ select • ←/→ category • Space toggle • A toggle all • Enter config • Esc back"
            }
            Stage::Configuration => "↑/↓ select • Enter toggle • S save • N next • Esc back",
            Stage::Confirm => "↑/↓ select • ←/→ adjust • Enter choose • Esc back",
            Stage::Installing => "L logs • Ctrl+R input mode • Enter send • Q recovery",
            Stage::Complete => "Esc recovery • B benchmarks",
            Stage::Benchmarks => "←/→ tabs • E export HTML • Esc/B back • Q quit",
            Stage::Recovery => "↑/↓ select • Enter apply • Q quit",
        }
    }

    fn spinner(&self) -> &'static str {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let idx = (self.tick_count % FRAMES.len() as u64) as usize;
        FRAMES[idx]
    }

    fn draw_welcome(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let mut lines = vec![
            Line::from(vec![Span::styled(
                "Welcome to Rusty-Stack",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from("High-performance AMD ML Stack installer"),
            Line::from(""),
            Line::from("• Press Enter to begin hardware detection"),
            Line::from("• Press Q to quit"),
        ];

        if self.entering_password {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Sudo password required to run installers",
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(
                "Type password and press Enter (input is hidden).",
            ));
            let masked = "*".repeat(self.password_input.len());
            lines.push(Line::from(format!("Password: {}", masked)));
        }

        let paragraph = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Welcome"))
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, area);
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

    fn draw_component_detection(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(4),
            ])
            .split(area);

        let visible: Vec<(usize, &Component)> = self
            .components
            .iter()
            .enumerate()
            .filter(|(_, component)| show_on_component_detection_screen(component))
            .collect();
        let total = visible.len();
        let checked = visible
            .iter()
            .filter(|(idx, _)| {
                let status = self
                    .component_detection_statuses
                    .get(*idx)
                    .copied()
                    .unwrap_or(ComponentDetectionStatus::Pending);
                matches!(
                    status,
                    ComponentDetectionStatus::Installed | ComponentDetectionStatus::NotInstalled
                )
            })
            .count();
        let installed = visible
            .iter()
            .filter(|(idx, _)| {
                self.component_detection_statuses.get(*idx).copied()
                    == Some(ComponentDetectionStatus::Installed)
            })
            .count();
        let not_installed = visible
            .iter()
            .filter(|(idx, _)| {
                self.component_detection_statuses.get(*idx).copied()
                    == Some(ComponentDetectionStatus::NotInstalled)
            })
            .count();
        let ratio = if total == 0 {
            1.0
        } else {
            checked as f64 / total as f64
        };

        let header = Paragraph::new(Text::from(vec![
            Line::from(vec![
                Span::styled(
                    "🔎 Component Detection",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(self.spinner(), Style::default().fg(Color::Cyan)),
                Span::raw(" Running comprehensive installed-component checks"),
            ]),
            Line::from(
                "Full probes stay enabled; slow ROCm/Python checks report progress instead of freezing.",
            ),
        ]))
        .block(Block::default().borders(Borders::ALL));
        frame.render_widget(header, chunks[0]);

        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Progress"))
            .gauge_style(Style::default().fg(Color::Green))
            .label(format!("{checked}/{total} checks complete"))
            .ratio(ratio);
        frame.render_widget(gauge, chunks[1]);

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(chunks[2]);

        let items: Vec<ListItem> = visible
            .iter()
            .map(|(idx, component)| {
                let status = self
                    .component_detection_statuses
                    .get(*idx)
                    .copied()
                    .unwrap_or(ComponentDetectionStatus::Pending);
                ListItem::new(Line::from(vec![
                    Span::styled(status.icon(), Style::default().fg(status.color())),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:<13}", status.label()),
                        Style::default().fg(status.color()),
                    ),
                    Span::raw(" "),
                    Span::raw(component.name.clone()),
                ]))
            })
            .collect();
        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Installable Component Checks"),
        );
        frame.render_widget(list, body[0]);

        let summary_prompt = if self.component_detection_done {
            "Complete. Press Enter to continue."
        } else {
            "Running. Please wait."
        };
        let summary = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "Detection Summary",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(format!("Checked: {checked}/{total}")),
            Line::from(format!("Installed: {installed}")),
            Line::from(format!("Not installed: {not_installed}")),
            Line::from(""),
            Line::from("Hidden here: verify/repair and benchmark actions."),
            Line::from("They remain available in their categories."),
            Line::from(""),
            Line::from(Span::styled(
                summary_prompt,
                Style::default().fg(if self.component_detection_done {
                    Color::Green
                } else {
                    Color::Yellow
                }),
            )),
        ]))
        .block(Block::default().borders(Borders::ALL).title("Summary"))
        .wrap(Wrap { trim: true });
        frame.render_widget(summary, body[1]);

        let footer = Paragraph::new(Text::from(vec![
            Line::from("This stage may import torch/vLLM/FastVideo and run ROCm probes."),
            Line::from("When detection finishes, press Enter to open component selection."),
        ]))
        .block(Block::default().borders(Borders::ALL).title("Status"));
        frame.render_widget(footer, chunks[3]);
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
            (Category::Core, "💠", "Core"),
            (Category::Extension, "📦", "Extensions"),
            (Category::UiUx, "🎨", "UI/UX"),
            (Category::Maintenance, "🛠️", "Maintenance"),
            (Category::Performance, "📊", "Performance"),
        ];
        let category_items: Vec<ListItem> = categories
            .iter()
            .enumerate()
            .map(|(idx, (_, icon, label))| {
                let style = if idx == self.selected_category {
                    Style::default()
                        .fg(Color::Yellow)
                        .bg(Color::DarkGray)
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
                    Category::Core => "💠",
                    Category::UiUx => "🎨",
                    Category::Extension => "📦",
                    Category::Maintenance => "🛠️",
                    Category::Performance => "📊",
                };

                let indicator = if comp.selected { "☑" } else { "☐" };
                let status_indicator = if comp.installed { "✓" } else { "○" };

                let display_name = if comp.experimental {
                    format!("🧪 {}", comp.name)
                } else {
                    comp.name.clone()
                };
                let line = format!(
                    "{} {} {} {} [{}]",
                    indicator, icon, display_name, status_indicator, comp.estimate
                );

                let style = if selected {
                    Style::default()
                        .fg(Color::Cyan)
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD)
                } else if comp.installed {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default()
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
            if comp.experimental {
                detail_lines.push(Line::from(Span::styled(
                    "⚠ EXPERIMENTAL BUILD",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )));
                if let Some(note) = &comp.note {
                    let mut wrap = String::new();
                    for word in note.split_whitespace() {
                        if !wrap.is_empty() && wrap.len() + word.len() + 1 > 42 {
                            detail_lines.push(Line::from(Span::styled(
                                std::mem::take(&mut wrap),
                                Style::default().fg(Color::Yellow),
                            )));
                        }
                        if !wrap.is_empty() {
                            wrap.push(' ');
                        }
                        wrap.push_str(word);
                    }
                    if !wrap.is_empty() {
                        detail_lines.push(Line::from(Span::styled(
                            wrap,
                            Style::default().fg(Color::Yellow),
                        )));
                    }
                }
            }
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
                Category::UiUx => {
                    if comp.installed {
                        ("Installed", Color::Green, "✓")
                    } else {
                        ("Not installed", Color::Yellow, "○")
                    }
                }
                Category::Maintenance => {
                    // Verify / repair are actions: "Completed" after a successful run.
                    if comp.installed {
                        ("Completed", Color::Green, "✓")
                    } else {
                        ("Not run", Color::Yellow, "○")
                    }
                }
                Category::Performance => {
                    if comp.installed {
                        ("Benchmarked", Color::Green, "✓")
                    } else if comp.progress > 0.0 {
                        // Was attempted but failed
                        ("Failed", Color::Red, "✗")
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
            } else if comp.category == Category::Maintenance {
                detail_lines.push(Line::from(""));
                detail_lines.push(Line::from(Span::styled(
                    "Maintenance Summary",
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
        let execution_mode = if self.config.batch_mode {
            "non-interactive"
        } else {
            "interactive"
        };
        let options = [
            format!("Execution Mode: {}", execution_mode),
            format!("Install Method: {}", self.config.install_method),
            "Back to Configuration".into(),
            self.action_label().into(),
        ];
        let mut lines = vec![
            Line::from(Span::styled(
                "Review your selection",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(format!("{}:", self.selection_label())),
            Line::from(""),
        ];
        for comp in selected {
            lines.push(Line::from(format!("• {} ({})", comp.name, comp.estimate)));
        }
        lines.push(Line::from(""));
        // Install-specific context — not meaningful for benchmark/verify runs.
        if self.run_mode() == RunMode::Install {
            lines.push(Line::from(format!(
                "ROCm Install Path: {}",
                self.config.install_path
            )));
        }
        lines.push(Line::from(format!("Execution Mode: {}", execution_mode)));
        if self.run_mode() == RunMode::Install {
            lines.push(Line::from(format!(
                "Install Method: {}",
                self.config.install_method
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            if self.run_mode() == RunMode::Install {
                "Pre-install options:"
            } else {
                "Run options:"
            },
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        for (idx, option) in options.iter().enumerate() {
            let style = if idx == self.confirm_selection {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let prefix = if idx == self.confirm_selection {
                "▶"
            } else {
                " "
            };
            lines.push(Line::from(Span::styled(
                format!("{} {}", prefix, option),
                style,
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from("Use Back to update config before install."));
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
            .constraints(
                [
                    Constraint::Length(3), // [0] progress gauge
                    Constraint::Length(2), // [1] install path
                    Constraint::Length(6), // [2] install details
                    Constraint::Min(5),    // [3] stage title / telemetry prompt
                    Constraint::Min(10),   // [4] body (log + sidebar)
                ]
                .as_ref(),
            )
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

        let install_path_text = Paragraph::new(Line::from(format!(
            "Install Path: {}",
            self.config.install_path
        )))
        .block(Block::default().borders(Borders::ALL).title("Target"));
        frame.render_widget(install_path_text, chunks[1]);

        let install_details = vec![
            Line::from(format!(
                "Target binary: {}/bin/llama-cli",
                self.config.install_path
            )),
            Line::from(format!("Install prefix: {}", self.config.install_path)),
        ];
        let install_details_panel = Paragraph::new(Text::from(install_details)).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Install Location"),
        );
        frame.render_widget(install_details_panel, chunks[2]);

        let stage_title = self.install_stage_title();
        let stage_title_area = if self.telemetry_prompt_pending {
            let prompt = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(2)].as_ref())
                .split(chunks[3]);
            let stage_panel = Paragraph::new(stage_title)
                .block(Block::default().borders(Borders::ALL).title("llama.cpp"));
            frame.render_widget(stage_panel, prompt[1]);
            Some(prompt[0])
        } else {
            let stage_panel = Paragraph::new(stage_title)
                .block(Block::default().borders(Borders::ALL).title("llama.cpp"));
            frame.render_widget(stage_panel, chunks[3]);
            None
        };

        if let Some(prompt_area) = stage_title_area {
            let prompt_text = Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    "Share anonymized build data to help validate future releases? [Y/n]",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from("Y/y/Enter submits build data • n/N skips silently"),
                Line::from("This prompt is part of the install flow."),
            ]))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Telemetry Opt-In"),
            )
            .wrap(Wrap { trim: true });
            frame.render_widget(prompt_text, prompt_area);
        }

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
            .split(chunks[4]);

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
                .title(format!("{} Log", self.flow_noun())),
        );
        frame.render_widget(log_list, body[0]);

        let selected: Vec<&Component> = self.components.iter().filter(|c| c.selected).collect();
        let installed_count = selected.iter().filter(|c| c.installed).count();
        let failed_count = selected
            .iter()
            .filter(|c| !c.installed && c.progress > 0.0)
            .count();
        let pending_count = selected.iter().filter(|c| c.progress == 0.0).count();
        let idle_secs = self.tick_count.saturating_sub(self.install_activity_tick) / 10;

        let base_lines = vec![
            Line::from(Span::styled(
                format!("{} Status", self.flow_noun()),
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
            Line::from(format!(
                "Activity: {} live; last output ~{}s ago",
                self.spinner(),
                idle_secs
            )),
        ];

        let tail_lines = vec![
            Line::from(""),
            Line::from(format!(
                "Input mode: {} (Ctrl+R to toggle)",
                self.install_input_mode.label()
            )),
            Line::from(format!("Input buffer: {}", self.install_input_buffer)),
            Line::from("Enter sends line; raw mode sends keystrokes"),
            Line::from(Span::styled(
                "Press L to open full logs",
                Style::default().fg(Color::Cyan),
            )),
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

        if self.install_log_popup {
            self.draw_install_log_popup(frame, area);
        }
    }

    fn draw_install_log_popup(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let width = area.width.saturating_sub(4).max(40);
        let height = area.height.saturating_sub(2).max(10);
        let popup = Rect {
            x: area.x + area.width.saturating_sub(width) / 2,
            y: area.y + area.height.saturating_sub(height) / 2,
            width,
            height,
        };

        frame.render_widget(Clear, popup);
        let file_hint = self
            .latest_install_log_path()
            .unwrap_or_else(|| "(not detected yet)".to_string());

        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled(
            format!("Full {} Logs", self.flow_noun()),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(format!("Log file: {}", file_hint)));
        lines.push(Line::from(""));

        let visible_lines = popup.height.saturating_sub(7) as usize;
        if self.logs.is_empty() {
            lines.push(Line::from("No logs captured yet."));
        } else {
            let max_scroll = self.logs.len().saturating_sub(visible_lines);
            let start = self.install_log_scroll.min(max_scroll);
            let end = (start + visible_lines).min(self.logs.len());
            for entry in &self.logs[start..end] {
                lines.push(Line::from(entry.clone()));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(format!(
                "Showing lines {}-{} of {}",
                start.saturating_add(1),
                end,
                self.logs.len()
            )));
        }
        lines.push(Line::from("Up/Down/PageUp/PageDown scroll • Esc/L close"));

        let widget = Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Logs")
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(widget, popup);
    }

    fn draw_complete(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        frame.render_widget(Clear, area);
        let (installed, failed, skipped) = self.partition_components();
        let (benchmarks, tests) = self.count_log_keywords();
        let mut lines = vec![
            Line::from(Span::styled(
                self.summary_title(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        lines.push(Line::from(Span::styled(
            format!("{}:", self.success_label()),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )));
        if installed.is_empty() {
            lines.push(Line::from("  (none)"));
        } else {
            for comp in installed.iter() {
                let (status, color) = if comp.installed {
                    (self.success_status(), Color::Green)
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

        // UI/UX Applications section with post-install instructions
        let uiux_apps: Vec<&Component> = self
            .components
            .iter()
            .filter(|c| c.category == Category::UiUx && c.installed)
            .collect();
        if !uiux_apps.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                Style::default().fg(Color::Magenta),
            )));
            lines.push(Line::from(Span::styled(
                " UI/UX Applications",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(Span::styled(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                Style::default().fg(Color::Magenta),
            )));

            for app in uiux_apps {
                lines.push(Line::from(""));
                if app.id == "vllm-studio" {
                    lines.push(Line::from(vec![Span::styled(
                        "🎨 vLLM Studio:",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )]));
                    lines.push(Line::from(vec![
                        Span::raw("   Run: "),
                        Span::styled("vllm-studio", Style::default().fg(Color::Green)),
                    ]));
                    lines.push(Line::from(vec![
                        Span::raw("   Tips: "),
                        Span::styled(
                            "Start from $HOME/vllm-studio/controller",
                            Style::default().fg(Color::Gray),
                        ),
                    ]));
                } else if app.id == "comfyui" {
                    lines.push(Line::from(vec![Span::styled(
                        "🎨 ComfyUI:",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )]));
                    lines.push(Line::from(vec![
                        Span::raw("   Run: "),
                        Span::styled("comfy", Style::default().fg(Color::Green)),
                    ]));
                    lines.push(Line::from(vec![
                        Span::raw("   URL: "),
                        Span::styled("http://localhost:8188", Style::default().fg(Color::Green)),
                    ]));
                    lines.push(Line::from(vec![
                        Span::raw("   Tips: "),
                        Span::styled(
                            "Uses ROCm GPU acceleration",
                            Style::default().fg(Color::Gray),
                        ),
                    ]));
                }
            }

            lines.push(Line::from(Span::styled(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                Style::default().fg(Color::Magenta),
            )));
        }

        let (env, verification) = self.partition_categories();
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "llama.cpp Turbo Quant",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from("Experimental tier • o"));
        if let Some(summary) = self.llama_completion_summary() {
            for line in summary {
                lines.push(line);
            }
        }
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

    fn llama_completion_summary(&self) -> Option<Vec<Line<'_>>> {
        let comp = self.components.iter().find(|c| c.id == "llama-cpp")?;
        if !comp.installed {
            return None;
        }
        Some(vec![
            Line::from(format!("Version: {}", self.llama_version_label())),
            Line::from(format!("Path: {}", self.config.install_path)),
            Line::from(format!("GPU Arch: {}", self.hardware.gpu.architecture)),
            Line::from(format!("Install Type: {}", self.llama_install_type_label())),
        ])
    }

    fn llama_version_label(&self) -> String {
        self.verification_reports
            .get("llama-cpp")
            .and_then(|lines| lines.iter().find(|line| line.contains("version")))
            .cloned()
            .unwrap_or_else(|| "installed".into())
    }

    fn llama_install_type_label(&self) -> &'static str {
        if self
            .logs
            .iter()
            .any(|line| line.contains("prebuilt") || line.contains("download"))
        {
            "pre-built"
        } else {
            "source"
        }
    }

    fn draw_benchmarks(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let results = load_benchmark_results();
        render_benchmark_page(frame, area, &results, self.benchmark_tab_index);
    }

    fn export_benchmark_report(&mut self) {
        let results = load_benchmark_results();
        match export_benchmark_report_html(&results, None) {
            Ok(path) => {
                let msg = format!("Benchmark HTML report exported: {}", path.display());
                self.logs.push(msg.clone());
                self.benchmark_notice = Some(UiNotice {
                    message: msg,
                    color: Color::Green,
                    expires_at_tick: self.tick_count.saturating_add(80),
                });
            }
            Err(err) => {
                let msg = format!("Failed to export benchmark HTML report: {}", err);
                self.errors.push(msg.clone());
                self.benchmark_notice = Some(UiNotice {
                    message: msg,
                    color: Color::Red,
                    expires_at_tick: self.tick_count.saturating_add(120),
                });
            }
        }
    }

    fn draw_notice(&self, frame: &mut Frame, area: Rect) {
        let Some(notice) = &self.benchmark_notice else {
            return;
        };

        let width = area.width.saturating_sub(4).clamp(20, 110);
        let height = 5u16;
        let x = area.x + area.width.saturating_sub(width) / 2;
        let y = area.y + 1;
        let popup = Rect {
            x,
            y,
            width,
            height,
        };

        frame.render_widget(Clear, popup);
        let text = vec![
            Line::from(Span::styled(
                "Benchmark Export",
                Style::default()
                    .fg(notice.color)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(notice.message.clone()),
        ];
        let notice_widget = Paragraph::new(Text::from(text))
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(notice.color)),
            );
        frame.render_widget(notice_widget, popup);
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
        let categories_len = 7i32; // Environment, Foundation, Core, Extension, UiUx, Verification, Performance
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
        // Flash Attention backends are mutually exclusive: both install the
        // identical `flash_attn` package, so the last-installed overwrites the
        // other (the .backend marker tracks which is active). Selecting one
        // deselects the other so the choice is explicit, never accidental.
        self.enforce_flash_attn_exclusivity(Some(index));
    }

    fn toggle_all(&mut self) {
        let any_selected = self.components.iter().any(|c| c.selected);
        for comp in self.components.iter_mut() {
            comp.selected = !any_selected;
        }
        // Bulk select may have picked both FA backends — keep at most one.
        self.enforce_flash_attn_exclusivity(None);
    }

    /// Ensure at most one Flash Attention backend is selected.
    /// - `Some(idx)`: the just-toggled component; only act if it's a FA backend
    ///   that was turned ON (then deselect the sibling).
    /// - `None` (bulk toggle): if more than one FA backend ended up selected,
    ///   keep the first in list order and deselect the rest.
    fn enforce_flash_attn_exclusivity(&mut self, toggled_index: Option<usize>) {
        const FA_BACKENDS: &[&str] = &["flash-attn-triton", "flash-attn-ck", "flash-attn"];
        match toggled_index {
            Some(idx) => {
                let turned_on = self
                    .components
                    .get(idx)
                    .map(|c| c.selected && FA_BACKENDS.contains(&c.id.as_str()))
                    == Some(true);
                if !turned_on {
                    return;
                }
                let toggled_id = self.components[idx].id.clone();
                for c in self.components.iter_mut() {
                    if c.id != toggled_id && FA_BACKENDS.contains(&c.id.as_str()) {
                        c.selected = false;
                    }
                }
            }
            None => {
                let first_selected = self
                    .components
                    .iter()
                    .position(|c| c.selected && FA_BACKENDS.contains(&c.id.as_str()));
                let Some(keep) = first_selected else {
                    return;
                };
                let keep_id = self.components[keep].id.clone();
                for c in self.components.iter_mut() {
                    if c.id != keep_id && FA_BACKENDS.contains(&c.id.as_str()) {
                        c.selected = false;
                    }
                }
            }
        }
    }

    fn config_rows(&self) -> Vec<(ConfigKey, String)> {
        vec![
            (
                ConfigKey::RocmPath,
                format!("ROCm Install Path: {}", self.config.install_path),
            ),
            (
                ConfigKey::BatchMode,
                format!("Batch Mode: {}", on_off(self.config.batch_mode)),
            ),
            (
                ConfigKey::AutoConfirm,
                format!("Auto Confirm: {}", on_off(self.config.auto_confirm)),
            ),
            (
                ConfigKey::StarRepo,
                format!("Star ML Stack Repo: {}", on_off(self.config.star_repos)),
            ),
            (
                ConfigKey::ForceReinstall,
                format!(
                    "Force Reinstall All: {}",
                    on_off(self.config.force_reinstall)
                ),
            ),
            (ConfigKey::Theme, format!("Theme: {}", self.config.theme)),
            (
                ConfigKey::PerfProfile,
                format!("Performance Profile: {}", self.config.performance_profile),
            ),
            (
                ConfigKey::VllmVersionLag,
                format!(
                    "vLLM version lag: {} ({}; 0=latest)",
                    self.config.vllm_version_lag,
                    if self.config.vllm_version_lag == 0 {
                        "latest"
                    } else {
                        "skip newest"
                    }
                ),
            ),
            (
                ConfigKey::VllmVersionAge,
                format!(
                    "vLLM min release age: {} days ({}; 0=off)",
                    self.config.vllm_version_min_age_days,
                    if self.config.vllm_version_min_age_days == 0 {
                        "off"
                    } else {
                        "age gate"
                    }
                ),
            ),
            (
                ConfigKey::OnnxInstallMethod,
                format!(
                    "ONNX install method: {} ({})",
                    self.config.onnx_install_method,
                    match self.config.onnx_install_method.as_str() {
                        "source" => "build from source",
                        "prebuilt" => "legacy onnxruntime-rocm",
                        _ => "PyPI onnxruntime-migraphx",
                    }
                ),
            ),
            (
                ConfigKey::OnnxVersion,
                format!(
                    "ONNX version: {}",
                    self.config
                        .onnx_version
                        .clone()
                        .unwrap_or_else(|| "default (1.25.0)".into())
                ),
            ),
            (ConfigKey::Save, "Save Configuration".into()),
        ]
    }

    fn selected_config_key(&self) -> ConfigKey {
        let mode = self.run_mode();
        self.config_rows()
            .into_iter()
            .filter(|(k, _)| config_applies(*k, mode))
            .nth(self.config_selection)
            .map(|(k, _)| k)
            .unwrap_or(ConfigKey::Save)
    }

    fn config_items(&self) -> Vec<String> {
        let mode = self.run_mode();
        self.config_rows()
            .into_iter()
            .filter(|(k, _)| config_applies(*k, mode))
            .map(|(_, label)| label)
            .collect()
    }

    fn config_help_lines(&self) -> Vec<Line<'_>> {
        let mut lines = match self.selected_config_key() {
            ConfigKey::RocmPath => vec![
                Line::from("ROCm install path used by installers."),
                Line::from("Default: /opt/rocm (system-wide ROCm)."),
            ],
            ConfigKey::BatchMode => vec![
                Line::from("Batch mode runs scripts non-interactively."),
                Line::from("Defaults are chosen when prompts appear."),
            ],
            ConfigKey::AutoConfirm => vec![
                Line::from("Auto confirm answers yes to prompts"),
                Line::from("when supported by the script."),
            ],
            ConfigKey::StarRepo => vec![
                Line::from("Star the ML Stack repository on GitHub."),
                Line::from("https://github.com/scooter-lacroix/Stan-s-ML-Stack"),
            ],
            ConfigKey::ForceReinstall => vec![
                Line::from("FORCE REINSTALL ALL COMPONENTS."),
                Line::from("Forces purging and re-downloading of everything."),
            ],
            ConfigKey::Theme => vec![
                Line::from("Theme affects TUI color styling."),
                Line::from("Switches between dark/light palettes."),
            ],
            ConfigKey::PerfProfile => vec![
                Line::from("Performance profile adjusts installer tuning."),
                Line::from("Balanced/performance/efficiency presets."),
            ],
            ConfigKey::VllmVersionLag => vec![
                Line::from("Supply-chain gate: skip this many newest vLLM releases."),
                Line::from("0=latest, 1=skip newest (default), 2+=more conservative."),
            ],
            ConfigKey::VllmVersionAge => vec![
                Line::from("Supply-chain gate: only adopt vLLM releases at least"),
                Line::from("this many days old (0=off). Combined with the lag."),
            ],
            ConfigKey::OnnxInstallMethod => vec![
                Line::from("ONNX Runtime install method."),
                Line::from("migraphx = PyPI onnxruntime-migraphx (default, working)."),
                Line::from("prebuilt = legacy onnxruntime-rocm (may mismatch ROCm)."),
                Line::from("source = build from source against /opt/rocm (heavy)."),
            ],
            ConfigKey::OnnxVersion => vec![
                Line::from("ONNX Runtime version override."),
                Line::from("default = pinned 1.25.0 (1.27.1 gives no benefit: the"),
                Line::from("`ort` crate's MIGraphX builder lacks model-cache in any release)."),
            ],
            ConfigKey::Save => vec![
                Line::from("Persist current settings to config.json."),
                Line::from("Dirty=yes means there are unsaved changes."),
            ],
        };
        lines.push(Line::from(""));
        lines.push(Line::from(TELEMETRY_PRIVACY_NOTE));
        lines
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

    fn move_confirm_selection(&mut self, delta: i32) {
        let len = 4i32;
        let mut idx = self.confirm_selection as i32 + delta;
        if idx < 0 {
            idx = len - 1;
        }
        if idx >= len {
            idx = 0;
        }
        self.confirm_selection = idx as usize;
    }

    fn adjust_confirm_selection(&mut self, delta: i32) {
        match self.confirm_selection {
            0 => {
                self.config.batch_mode = if delta < 0 {
                    true
                } else if delta > 0 {
                    false
                } else {
                    !self.config.batch_mode
                };
                self.config_dirty = true;
            }
            1 => {
                self.cycle_install_method(delta);
                self.config_dirty = true;
            }
            _ => {}
        }
    }

    fn activate_confirm_selection(&mut self) {
        match self.confirm_selection {
            0 => self.adjust_confirm_selection(0),
            1 => self.adjust_confirm_selection(1),
            2 => self.stage = Stage::Configuration,
            3 => self.start_installation(),
            _ => {}
        }
    }

    fn cycle_install_method(&mut self, delta: i32) {
        let methods = ["auto", "global", "venv"];
        let current = methods
            .iter()
            .position(|m| *m == self.config.install_method)
            .unwrap_or(0);
        let next = (current as i32 + delta).rem_euclid(methods.len() as i32) as usize;
        self.config.install_method = methods[next].into();
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
        match self.selected_config_key() {
            ConfigKey::BatchMode => {
                self.config.batch_mode = !self.config.batch_mode;
                self.config_dirty = true;
            }
            ConfigKey::AutoConfirm => {
                self.config.auto_confirm = !self.config.auto_confirm;
                self.config_dirty = true;
            }
            ConfigKey::StarRepo => {
                self.config.star_repos = !self.config.star_repos;
                self.config_dirty = true;
            }
            ConfigKey::ForceReinstall => {
                self.config.force_reinstall = !self.config.force_reinstall;
                self.config_dirty = true;
            }
            ConfigKey::Theme => {
                self.config.theme = if self.config.theme == "dark" {
                    "light".into()
                } else {
                    "dark".into()
                };
                self.config_dirty = true;
            }
            ConfigKey::PerfProfile => {
                let profiles = ["balanced", "performance", "efficiency"];
                let current = profiles
                    .iter()
                    .position(|p| *p == self.config.performance_profile)
                    .unwrap_or(0);
                let next = (current + 1) % profiles.len();
                self.config.performance_profile = profiles[next].into();
                self.config_dirty = true;
            }
            ConfigKey::VllmVersionLag => {
                // cycle 0 → 1 → 2 → 3 → 4 → 5 → 0
                self.config.vllm_version_lag = (self.config.vllm_version_lag + 1) % 6;
                self.config_dirty = true;
            }
            ConfigKey::VllmVersionAge => {
                // cycle off + common age windows (days)
                let ages = [0, 7, 14, 30, 60, 90];
                let cur = ages
                    .iter()
                    .position(|&a| a == self.config.vllm_version_min_age_days)
                    .unwrap_or(0);
                self.config.vllm_version_min_age_days = ages[(cur + 1) % ages.len()];
                self.config_dirty = true;
            }
            ConfigKey::OnnxInstallMethod => {
                // cycle migraphx → prebuilt → source → migraphx
                let methods = ["migraphx", "prebuilt", "source"];
                let cur = methods
                    .iter()
                    .position(|m| *m == self.config.onnx_install_method)
                    .unwrap_or(0);
                self.config.onnx_install_method = methods[(cur + 1) % methods.len()].into();
                self.config_dirty = true;
            }
            ConfigKey::OnnxVersion => {
                // cycle default → 1.25.0 → 1.27.1 → default
                let cur = self.config.onnx_version.as_deref();
                let next = match cur {
                    None => Some("1.25.0"),
                    Some("1.25.0") => Some("1.27.1"),
                    _ => None, // back to default (pinned)
                };
                self.config.onnx_version = next.map(str::to_string);
                self.config_dirty = true;
            }
            ConfigKey::Save => self.save_config(),
            ConfigKey::RocmPath => {} // display-only, no toggle
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
                self.start_component_status_detection();
            }
            3 => self.should_exit = true,
            _ => {}
        }
    }

    fn start_installation(&mut self) {
        let selected = self.selected_components();
        if selected.is_empty() {
            self.errors
                .push("No components selected. Go back and select at least one component.".into());
            self.stage = Stage::Recovery;
            return;
        }
        // Only require sudo password if at least one selected component needs it
        let any_needs_sudo = selected.iter().any(|c| c.needs_sudo);
        #[cfg(unix)]
        let is_non_root = unsafe { libc::geteuid() != 0 };
        #[cfg(not(unix))]
        let is_non_root = false;
        if is_non_root && any_needs_sudo && self.sudo_password.is_none() {
            self.errors
                .push("Sudo password required for selected components. Press Esc to go back to Welcome screen and enter your sudo password.".into());
            self.stage = Stage::Recovery;
            return;
        }

        // Set FORCE=true if force_reinstall is enabled
        if self.config.force_reinstall {
            std::env::set_var("FORCE", "true");
            std::env::set_var("PYTORCH_REINSTALL", "true");
            std::env::set_var("MLSTACK_FORCE_REINSTALL", "true");
        } else {
            std::env::remove_var("FORCE");
            std::env::remove_var("PYTORCH_REINSTALL");
            std::env::remove_var("MLSTACK_FORCE_REINSTALL");
        }

        // ONNX install options (TUI config → env → dispatch). The dispatch reads
        // MLSTACK_ONNX_INSTALL_METHOD / MLSTACK_ONNX_VERSION. Defaults (migraphx /
        // pinned 1.25.0) match the dispatch defaults, so this only changes behavior
        // when the user cycles them on the Configuration screen.
        std::env::set_var(
            "MLSTACK_ONNX_INSTALL_METHOD",
            &self.config.onnx_install_method,
        );
        match &self.config.onnx_version {
            Some(v) => std::env::set_var("MLSTACK_ONNX_VERSION", v),
            None => std::env::remove_var("MLSTACK_ONNX_VERSION"),
        }

        self.stage = Stage::Installing;
        self.install_status.progress = 0.0;
        self.install_status.message = "Starting installation".into();
        self.mark_install_activity();
        let (tx, rx) = mpsc::channel();
        let (input_tx, input_rx) = mpsc::channel();
        let sudo_password = self.sudo_password.clone();
        let config = self.config.clone();
        self.install_input_sender = Some(input_tx);
        self.install_input_buffer.clear();
        self.install_log_popup = false;
        self.install_log_scroll = 0;
        self.install_log_file_path = None;
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
                InstallerEvent::Log(line, is_transient) => {
                    self.mark_install_activity();
                    if let Some(progress) = parse_log_progress(&line) {
                        self.apply_log_progress(progress, &line);
                    }
                    self.push_log_ext(line, is_transient);
                }
                InstallerEvent::Progress {
                    component_id,
                    progress,
                    message,
                } => {
                    self.mark_install_activity();
                    if component_id == "__overall__" {
                        self.install_status.progress = progress.clamp(0.0, 1.0);
                    } else {
                        self.update_component_progress(&component_id, progress);
                        self.recalculate_overall_progress();
                    }
                    self.install_status.message = message.clone();
                    // Progress messages are usually transient
                    self.push_log_ext(message, true);
                }
                InstallerEvent::ComponentStart { component_id, name } => {
                    self.mark_install_activity();
                    if let Some(comp) = self.components.iter_mut().find(|c| c.id == component_id) {
                        self.install_status.message = if comp.category == Category::Performance {
                            format!("Running {}", name)
                        } else {
                            format!("Installing {}", name)
                        };
                        comp.progress = 0.05;
                        comp.installed = false;
                    } else {
                        self.install_status.message = format!("Installing {}", name);
                    }
                    self.recalculate_overall_progress();
                    self.push_log(self.install_status.message.clone());
                }
                InstallerEvent::ComponentComplete {
                    component_id,
                    success,
                    message,
                } => {
                    self.mark_install_activity();
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
                    self.mark_install_activity();
                    let cleaned = lines
                        .into_iter()
                        .map(|line| Self::sanitize_line(&line))
                        .collect();
                    self.verification_reports.insert(component_id, cleaned);
                }
                InstallerEvent::Finished { success } => {
                    self.mark_install_activity();
                    self.install_status.completed = success;
                    self.install_input_sender = None;
                    self.install_input_buffer.clear();
                    self.recalculate_overall_progress();
                    self.install_status.progress = 1.0;
                    if success {
                        // Stay in Installing to present the telemetry opt-in prompt.
                        // The prompt is rendered by draw_installing() and resolved in
                        // handle_key() (Y/n/Enter) which transitions to Benchmarks.
                        self.telemetry_prompt_pending = true;
                        self.telemetry_prompt_active = true;
                    } else {
                        self.stage = Stage::Complete;
                    }
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

    fn apply_log_progress(&mut self, progress: f32, line: &str) {
        let next = progress.clamp(0.0, 1.0);
        if let Some(component) = self
            .components
            .iter_mut()
            .find(|c| c.selected && c.progress > 0.0 && c.progress < 1.0)
        {
            if next > component.progress {
                component.progress = next;
            }
            self.recalculate_overall_progress();
        } else if next > self.install_status.progress {
            self.install_status.progress = next;
        }

        let percent = (next * 100.0).round() as i32;
        self.install_status.message =
            format!("Build activity: {percent}% {}", trim_status_line(line));
    }

    fn mark_install_activity(&mut self) {
        self.install_activity_tick = self.tick_count;
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
        if component.category == Category::Maintenance {
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

            if comp.category != Category::Maintenance || !comp.id.starts_with("verify-") {
                let task_label = if comp.category == Category::Performance {
                    "Benchmark"
                } else if comp.category == Category::Maintenance {
                    "Repair"
                } else {
                    "Install"
                };
                lines.push(self.task_line(task_label, self.install_task_status(comp)));
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

    fn install_stage_title(&self) -> Text<'_> {
        let lines = match self.install_status.message.as_str() {
            m if m.contains("download") => vec![
                Line::from("Pre-built download in progress"),
                Line::from("Progress is tied to download completion."),
            ],
            m if m.contains("configure") => vec![
                Line::from("Source build stage: configure"),
                Line::from(self.llama_build_flags_line()),
            ],
            m if m.contains("compile") => vec![
                Line::from("Source build stage: compile"),
                Line::from(self.llama_build_flags_line()),
            ],
            m if m.contains("link") => vec![
                Line::from("Source build stage: link"),
                Line::from(self.llama_build_flags_line()),
            ],
            _ => vec![
                Line::from("Installing llama.cpp Turbo Quant"),
                Line::from(self.llama_build_flags_line()),
            ],
        };
        let mut lines = lines;
        lines.push(Line::from(""));
        lines.push(Line::from(self.telemetry_inline_prompt()));
        Text::from(lines)
    }

    fn llama_build_flags_line(&self) -> String {
        let config = LlamaCppConfig {
            gpu_arch: self.hardware.gpu.architecture.clone(),
            ..Default::default()
        };
        let flags = LlamaCppInstaller::new(config).cmake_flags().join(" ");
        format!("Build flags: {}", flags)
    }

    fn telemetry_inline_prompt(&self) -> String {
        let status = self
            .telemetry_gate
            .as_ref()
            .map(|gate| {
                if gate.is_enabled() {
                    TELEMETRY_STATUS_ENABLED
                } else {
                    TELEMETRY_STATUS_DISABLED
                }
            })
            .unwrap_or(TELEMETRY_STATUS_DISABLED);
        format!(
            "{} | {} / {} | {}",
            TELEMETRY_DESCRIPTION, TELEMETRY_ENABLE_LABEL, TELEMETRY_DISABLE_LABEL, status
        )
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

    fn start_component_status_detection(&mut self) {
        let (tx, rx) = mpsc::channel();
        let mut components = self.components.clone();
        let force_reinstall = self.config.force_reinstall;

        self.component_detection_statuses = components
            .iter()
            .map(|component| {
                if component.category == Category::Maintenance {
                    ComponentDetectionStatus::Skipped
                } else {
                    ComponentDetectionStatus::Pending
                }
            })
            .collect();
        self.component_detection_done = false;
        self.component_status_receiver = Some(rx);
        self.stage = Stage::ComponentDetect;

        thread::spawn(move || {
            let python_candidates = python_interpreters();
            for (index, component) in components.iter_mut().enumerate() {
                if component.category == Category::Maintenance {
                    let _ = tx.send(ComponentStatusEvent::Skipped(index));
                    continue;
                }

                let _ = tx.send(ComponentStatusEvent::Started(index));
                component.installed = is_component_installed(component, &python_candidates);
                if component.installed && !force_reinstall {
                    // Only auto-deselect when force reinstall is OFF.
                    // When force reinstall is ON, keep installed components selected
                    // so the installer will properly purge and reinstall them.
                    component.selected = false;
                }
                let _ = tx.send(ComponentStatusEvent::Finished {
                    index,
                    installed: component.installed,
                });
            }
            let _ = tx.send(ComponentStatusEvent::Complete(components));
        });
    }

    fn poll_component_status_detection(&mut self) {
        let events: Vec<_> = if let Some(receiver) = &self.component_status_receiver {
            std::iter::from_fn(|| receiver.try_recv().ok()).collect()
        } else {
            Vec::new()
        };

        for event in events {
            match event {
                ComponentStatusEvent::Started(index) => {
                    if let Some(status) = self.component_detection_statuses.get_mut(index) {
                        *status = ComponentDetectionStatus::Running;
                    }
                }
                ComponentStatusEvent::Finished { index, installed } => {
                    if let Some(status) = self.component_detection_statuses.get_mut(index) {
                        *status = if installed {
                            ComponentDetectionStatus::Installed
                        } else {
                            ComponentDetectionStatus::NotInstalled
                        };
                    }
                }
                ComponentStatusEvent::Skipped(index) => {
                    if let Some(status) = self.component_detection_statuses.get_mut(index) {
                        *status = ComponentDetectionStatus::Skipped;
                    }
                }
                ComponentStatusEvent::Complete(components) => {
                    self.components = components;
                    self.component_status_receiver = None;
                    self.component_detection_done = true;
                }
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

    /// Derive the run mode from the selected components so the flow verbiage
    /// matches what is actually happening. A pure-Performance selection runs
    /// benchmarks; pure verify-* Maintenance selection runs verification;
    /// repair actions and installable components run as install/action flows.
    fn run_mode(&self) -> RunMode {
        let selected: Vec<&Component> = self.components.iter().filter(|c| c.selected).collect();
        if selected.is_empty() {
            return RunMode::Install;
        }
        if selected.iter().all(|c| c.category == Category::Performance) {
            RunMode::Benchmark
        } else if selected
            .iter()
            .all(|c| c.category == Category::Maintenance && c.id.starts_with("verify-"))
        {
            RunMode::Verify
        } else {
            RunMode::Install
        }
    }

    /// "Installation" / "Benchmark" / "Verification" — the flow noun for titles.
    fn flow_noun(&self) -> &'static str {
        match self.run_mode() {
            RunMode::Install => "Installation",
            RunMode::Benchmark => "Benchmark",
            RunMode::Verify => "Verification",
        }
    }

    /// The primary action button label on the confirm screen.
    fn action_label(&self) -> &'static str {
        match self.run_mode() {
            RunMode::Install => "Start Installation",
            RunMode::Benchmark => "Run Benchmarks",
            RunMode::Verify => "Run Verification",
        }
    }

    /// Heading for the selected-components list.
    fn selection_label(&self) -> &'static str {
        match self.run_mode() {
            RunMode::Install => "Selected Components",
            RunMode::Benchmark => "Selected Benchmarks",
            RunMode::Verify => "Selected Checks",
        }
    }

    /// The summary screen title.
    fn summary_title(&self) -> &'static str {
        match self.run_mode() {
            RunMode::Install => "Installation Summary",
            RunMode::Benchmark => "Benchmark Results",
            RunMode::Verify => "Verification Summary",
        }
    }

    /// Noun for a component that completed successfully on the summary screen.
    fn success_label(&self) -> &'static str {
        match self.run_mode() {
            RunMode::Install => "Installed",
            RunMode::Benchmark => "Completed",
            RunMode::Verify => "Passed",
        }
    }

    /// Lowercase status word shown per-component on the summary screen.
    fn success_status(&self) -> &'static str {
        match self.run_mode() {
            RunMode::Install => "installed",
            RunMode::Benchmark => "completed",
            RunMode::Verify => "passed",
        }
    }

    fn current_category(&self) -> Category {
        match self.selected_category {
            0 => Category::Environment,
            1 => Category::Foundation,
            2 => Category::Core,
            3 => Category::Extension,
            4 => Category::UiUx,
            5 => Category::Maintenance,
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
        if let Some(path) = clean_line.split("Log file:").nth(1) {
            let candidate = path.trim();
            if !candidate.is_empty() {
                self.install_log_file_path = Some(candidate.to_string());
            }
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

    fn latest_install_log_path(&self) -> Option<String> {
        if let Some(path) = &self.install_log_file_path {
            return Some(path.clone());
        }
        if self.config.log_dir.trim().is_empty() {
            return None;
        }
        Some(
            std::path::Path::new(&self.config.log_dir)
                .join("rusty-stack.log")
                .display()
                .to_string(),
        )
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
            .filter(|c| c.category == Category::Maintenance && c.selected)
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
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                // Handle both direct exports and if-guarded exports
                if let Some(export_idx) = trimmed.rfind("export ") {
                    let rest = &trimmed[export_idx + 7..]; // skip "export "
                    if let Some((key, value)) = rest.split_once('=') {
                        let key = key.trim();
                        if key.is_empty() || key.contains(' ') || key.contains('$') {
                            continue;
                        }
                        let value = value.trim();
                        // Strip shell suffixes (; fi, &&, etc.) from value
                        let value = Self::strip_shell_value(value);
                        let value = value
                            .strip_prefix('"')
                            .and_then(|v| v.strip_suffix('"'))
                            .unwrap_or(&value);
                        entries.push((key.to_string(), value.to_string()));
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

    /// Strip shell command suffixes from an env file value.
    ///
    /// The `.mlstack_env` file generated by `env_setup.rs` uses if-guard patterns:
    /// `if [ -z "${VAR:-}" ]; then export VAR=value; fi`
    ///
    /// When parsing the `export VAR=value` part from inside the if-guard,
    /// the value may contain the trailing `; fi`. This function strips it.
    fn strip_shell_value(value: &str) -> String {
        let mut v = value.to_string();
        // Strip trailing "; fi" (from if-guard lines)
        if let Some(idx) = v.find("; fi") {
            v.truncate(idx);
        }
        // Strip trailing "&& ..." (from chained commands)
        if let Some(idx) = v.find("&& ") {
            v.truncate(idx);
        }
        v.trim().to_string()
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

/// Stable identity for a config row, independent of which rows the current
/// run-mode filters into the visible list. Toggle/help key off this, not list
/// position, so hiding install-only rows for benchmark/verify can't desync the
/// action from the highlighted row.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConfigKey {
    RocmPath,
    BatchMode,
    AutoConfirm,
    StarRepo,
    ForceReinstall,
    Theme,
    PerfProfile,
    VllmVersionLag,
    VllmVersionAge,
    OnnxInstallMethod,
    OnnxVersion,
    Save,
}

/// Which config rows apply to the current run mode. Benchmark/verify runs hide
/// install-only rows (ROCm path, star repo, force reinstall, perf profile)
/// since they aren't meaningful for those runs.
fn config_applies(key: ConfigKey, mode: RunMode) -> bool {
    match mode {
        RunMode::Install => true,
        RunMode::Benchmark | RunMode::Verify => matches!(
            key,
            ConfigKey::BatchMode | ConfigKey::AutoConfirm | ConfigKey::Theme | ConfigKey::Save
        ),
    }
}

fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

fn parse_log_progress(line: &str) -> Option<f32> {
    let open = line.find('[')?;
    let percent = line[open + 1..].find('%')? + open + 1;
    let value = line[open + 1..percent].trim().parse::<f32>().ok()?;
    Some((value / 100.0).clamp(0.0, 1.0))
}

fn trim_status_line(line: &str) -> String {
    const MAX: usize = 72;
    let trimmed = line.trim();
    if trimmed.chars().count() <= MAX {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(MAX.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn show_on_component_detection_screen(component: &Component) -> bool {
    !matches!(
        component.category,
        Category::Maintenance | Category::Performance
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn preflight_enter_starts_component_detection_screen() {
        let mut app = App::new("/tmp".into());
        app.entering_password = false;
        app.components.clear();
        app.stage = Stage::Preflight;
        app.preflight.can_continue = true;

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.stage, Stage::ComponentDetect);
        assert!(app.component_status_receiver.is_some());
    }

    #[test]
    fn component_detection_screen_hides_action_categories() {
        let maintenance = Component {
            id: "rccl-repair".into(),
            name: "Repair RCCL Multi-GPU".into(),
            description: String::new(),
            script: String::new(),
            category: Category::Maintenance,
            required: false,
            selected: false,
            installed: false,
            progress: 0.0,
            estimate: String::new(),
            needs_sudo: false,
            experimental: false,
            note: None,
        };
        let mut performance = maintenance.clone();
        performance.id = "all-benchmarks".into();
        performance.name = "Full Suite Benchmark".into();
        performance.category = Category::Performance;
        let mut installable = maintenance.clone();
        installable.id = "fastvideo".into();
        installable.name = "FastVideo".into();
        installable.category = Category::Extension;

        assert!(!show_on_component_detection_screen(&maintenance));
        assert!(!show_on_component_detection_screen(&performance));
        assert!(show_on_component_detection_screen(&installable));
    }

    #[test]
    fn explicit_overall_progress_event_updates_overall_bar() {
        let mut app = App::new("/tmp".into());
        app.entering_password = false;
        app.components.clear();
        app.components.push(Component {
            id: "rccl-repair".into(),
            name: "Repair RCCL Multi-GPU".into(),
            description: String::new(),
            script: String::new(),
            category: Category::Maintenance,
            required: false,
            selected: true,
            installed: false,
            progress: 0.05,
            estimate: String::new(),
            needs_sudo: false,
            experimental: false,
            note: None,
        });
        let (tx, rx) = mpsc::channel();
        app.install_receiver = Some(rx);

        tx.send(InstallerEvent::Progress {
            component_id: "__overall__".into(),
            progress: 0.73,
            message: "RCCL build: 73%".into(),
        })
        .unwrap();
        app.poll_installer();

        assert!((app.install_status.progress - 0.73).abs() < f32::EPSILON);
        assert_eq!(app.install_status.message, "RCCL build: 73%");
    }

    #[test]
    fn build_log_percentage_updates_running_component_progress() {
        let mut app = App::new("/tmp".into());
        app.entering_password = false;
        app.components.clear();
        app.components.push(Component {
            id: "rccl-repair".into(),
            name: "Repair RCCL Multi-GPU".into(),
            description: String::new(),
            script: String::new(),
            category: Category::Maintenance,
            required: false,
            selected: true,
            installed: false,
            progress: 0.05,
            estimate: String::new(),
            needs_sudo: false,
            experimental: false,
            note: None,
        });
        let (tx, rx) = mpsc::channel();
        app.install_receiver = Some(rx);

        tx.send(InstallerEvent::Log(
            "[ 70%] Building CXX object CMakeFiles/rccl.dir/foo.cpp.o".into(),
            true,
        ))
        .unwrap();
        app.poll_installer();

        assert!((app.components[0].progress - 0.70).abs() < f32::EPSILON);
        assert!((app.install_status.progress - 0.70).abs() < f32::EPSILON);
        assert!(app.install_status.message.contains("70%"));
    }

    #[test]
    fn component_detection_complete_waits_for_enter() {
        let mut app = App::new("/tmp".into());
        app.entering_password = false;
        app.components.clear();
        app.stage = Stage::ComponentDetect;
        let (tx, rx) = mpsc::channel();
        app.component_status_receiver = Some(rx);

        tx.send(ComponentStatusEvent::Complete(Vec::new())).unwrap();
        app.poll_component_status_detection();

        assert_eq!(app.stage, Stage::ComponentDetect);

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.stage, Stage::ComponentSelect);
    }
}
