//! Screens
//!
//! Different screen layouts for the TUI with Rusty Stack branding.

use anyhow::Result;
use crossterm::event::KeyEvent;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table, Tabs, Wrap};
use std::sync::mpsc::{self, Receiver};
use std::thread;

/// Screen trait for different TUI screens.
pub trait Screen {
    /// Draws the screen.
    fn draw(&self, frame: &mut ratatui::Frame, area: Rect);
    /// Handles keyboard input.
    fn handle_key(&mut self, key: KeyEvent) -> Result<()>;
    /// Updates the screen (called every tick).
    fn update(&mut self);
    /// Goes to next tab.
    fn next_tab(&mut self);
    /// Goes to previous tab.
    fn prev_tab(&mut self);
    /// Sets the current tab.
    fn set_tab(&mut self, tab: usize);
    /// Checks if the screen needs to be redrawn.
    fn is_dirty(&self) -> bool {
        false
    }
    /// Marks the screen as clean (no longer needs redraw).
    fn mark_clean(&mut self) {}
}

/// Component status for installation
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    pub name: String,
    pub description: String,
    pub installed: bool,
    pub selected: bool,
    pub version: Option<String>,
}

impl ComponentInfo {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            installed: false,
            selected: false,
            version: None,
        }
    }
}

/// Main dashboard screen with tabs.
pub struct MainScreen {
    /// Current tab
    current_tab: usize,
    /// Tab names
    tabs: Vec<String>,
    /// Overview data
    overview_data: OverviewData,
    /// Component list for Install tab
    components: Vec<ComponentInfo>,
    /// Selected component index
    selected_component: usize,
    /// Verification results
    verify_results: Vec<(String, bool, String)>,
    /// Settings
    settings: Vec<(String, String, String)>,
    /// Selected setting index
    selected_setting: usize,
    /// Verification result receiver
    verify_receiver: Option<Receiver<Vec<(String, bool, String)>>>,
    /// Verification in progress
    verifying: bool,
    /// Dirty flag for redraw optimization
    dirty: bool,
}

/// Overview tab data
#[derive(Default)]
pub struct OverviewData {
    pub gpu_detected: bool,
    pub rocm_installed: bool,
    pub pytorch_installed: bool,
    pub triton_installed: bool,
    pub vllm_installed: bool,
}

impl MainScreen {
    /// Creates a new main screen.
    pub fn new() -> Self {
        let components = vec![
            // Foundation
            ComponentInfo::new("ROCm", "AMD ROCm platform for GPU computing"),
            ComponentInfo::new("PyTorch", "Deep learning framework with ROCm support"),
            ComponentInfo::new("Triton", "OpenAI Triton compiler for AMD GPUs"),
            // Inference & Optimization
            ComponentInfo::new("vLLM", "High-throughput LLM inference engine"),
            ComponentInfo::new("ONNX Runtime", "Cross-platform inference accelerator"),
            ComponentInfo::new("MIGraphX", "AMD graph optimization library"),
            ComponentInfo::new("Flash Attention", "Efficient attention for transformers"),
            // Training & Optimization
            ComponentInfo::new("DeepSpeed", "Distributed training with ZeRO optimization"),
            ComponentInfo::new("Megatron-LM", "Large-scale training framework"),
            ComponentInfo::new("BITSANDBYTES", "8-bit/4-bit quantization for LLMs"),
            ComponentInfo::new("AITER", "AMD AI Tensor Engine for RDNA 3 GPUs"),
            // Monitoring & Profiling
            ComponentInfo::new("ROCm SMI", "GPU monitoring and management"),
            ComponentInfo::new("PyTorch Profiler", "Performance profiling with TensorBoard"),
            ComponentInfo::new("Weights & Biases", "ML experiment tracking and visualization"),
        ];

        let verify_results = vec![
            // Foundation
            ("ROCm".to_string(), false, "Not checked".to_string()),
            ("PyTorch".to_string(), false, "Not checked".to_string()),
            ("Triton".to_string(), false, "Not checked".to_string()),
            // Inference & Optimization
            ("vLLM".to_string(), false, "Not checked".to_string()),
            ("ONNX Runtime".to_string(), false, "Not checked".to_string()),
            ("MIGraphX".to_string(), false, "Not checked".to_string()),
            ("Flash Attention".to_string(), false, "Not checked".to_string()),
            // Training & Optimization
            ("DeepSpeed".to_string(), false, "Not checked".to_string()),
            ("Megatron-LM".to_string(), false, "Not checked".to_string()),
            ("BitsAndBytes".to_string(), false, "Not checked".to_string()),
            ("AITER".to_string(), false, "Not checked".to_string()),
            // Monitoring & Profiling
            ("ROCm SMI".to_string(), false, "Not checked".to_string()),
            ("PyTorch Profiler".to_string(), false, "Not checked".to_string()),
            ("Weights & Biases".to_string(), false, "Not checked".to_string()),
            // System checks
            ("GPU Access".to_string(), false, "Not checked".to_string()),
            ("CUDA Compat".to_string(), false, "Not checked".to_string()),
        ];

        let settings = vec![
            ("ROCm Path".to_string(), "/opt/rocm".to_string(), "Installation path for ROCm".to_string()),
            ("GPU Architecture".to_string(), "gfx1100".to_string(), "Target GPU architecture".to_string()),
            ("Batch Mode".to_string(), "off".to_string(), "Non-interactive installation".to_string()),
            ("Log Level".to_string(), "info".to_string(), "Logging verbosity".to_string()),
        ];

        Self {
            current_tab: 0,
            tabs: vec![
                "Overview".to_string(),
                "Install".to_string(),
                "Verify".to_string(),
                "Settings".to_string(),
            ],
            overview_data: OverviewData::default(),
            components,
            selected_component: 0,
            verify_results,
            settings,
            selected_setting: 0,
            verify_receiver: None,
            verifying: false,
            dirty: true,
        }
    }

    /// Marks the screen as dirty (needs redraw).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Checks if the screen needs to be redrawn.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Marks the screen as clean (no longer needs redraw).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Starts verification in background.
    pub fn start_verification(&mut self) {
        if self.verifying {
            return; // Already running
        }

        self.verifying = true;
        self.update_verify_status("Verifying...", false);

        let (tx, rx) = mpsc::channel();
        self.verify_receiver = Some(rx);

        thread::spawn(move || {
            use mlstack_installers::verification::VerificationModule;

            let rt = tokio::runtime::Runtime::new().unwrap();
            let results = rt.block_on(async {
                VerificationModule::run_all().await
            });

            // Convert to simple format for TUI
            let simple_results: Vec<(String, bool, String)> = results.into_iter()
                .map(|item| {
                    let passed = matches!(item.status,
                        mlstack_installers::verification::VerificationStatus::Success |
                        mlstack_installers::verification::VerificationStatus::Warning
                    );
                    (item.name, passed, item.message.unwrap_or_default())
                })
                .collect();

            let _ = tx.send(simple_results);
        });
    }

    /// Polls for verification results.
    pub fn poll_verification(&mut self) {
        if let Some(rx) = &self.verify_receiver {
            if let Ok(results) = rx.try_recv() {
                self.verify_results = results;
                self.verify_receiver = None;
                self.verifying = false;
                self.dirty = true; // Mark dirty to show new results
            }
        }
    }

    /// Updates the verification status display.
    fn update_verify_status(&mut self, message: &str, _passed: bool) {
        // Update all verify results to show "verifying" state
        for result in &mut self.verify_results {
            result.1 = false;
            result.2 = message.to_string();
        }
    }

    /// Draws the overview tab.
    fn draw_overview(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);

        // Left panel: Welcome and quick status
        let welcome_lines = vec![
            Line::from(vec![
                Span::styled("Welcome to ", Style::default().fg(Color::White)),
                Span::styled("Rusty-Stack", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("High-performance AMD ML Stack installer", Style::default().fg(Color::Gray)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Quick Actions:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  1 ", Style::default().fg(Color::Cyan)),
                Span::raw("Overview (this screen)"),
            ]),
            Line::from(vec![
                Span::styled("  2 ", Style::default().fg(Color::Cyan)),
                Span::raw("Install components"),
            ]),
            Line::from(vec![
                Span::styled("  3 ", Style::default().fg(Color::Cyan)),
                Span::raw("Verify installation"),
            ]),
            Line::from(vec![
                Span::styled("  4 ", Style::default().fg(Color::Cyan)),
                Span::raw("Settings"),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  R ", Style::default().fg(Color::Cyan)),
                Span::raw("Refresh hardware detection"),
            ]),
            Line::from(vec![
                Span::styled("  Tab ", Style::default().fg(Color::Cyan)),
                Span::raw("Switch tabs"),
            ]),
            Line::from(vec![
                Span::styled("  Q ", Style::default().fg(Color::Cyan)),
                Span::raw("Quit"),
            ]),
        ];

        let welcome = Paragraph::new(Text::from(welcome_lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(" Welcome ", Style::default().fg(Color::Cyan)))
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(welcome, chunks[0]);

        // Right panel: Quick status
        let status_icon = |installed: bool| -> Span {
            if installed {
                Span::styled("✓", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
            } else {
                Span::styled("○", Style::default().fg(Color::Yellow))
            }
        };

        let status_lines = vec![
            Line::from(vec![
                Span::styled("Component Status", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                status_icon(self.overview_data.gpu_detected),
                Span::raw(" GPU Detected"),
            ]),
            Line::from(vec![
                status_icon(self.overview_data.rocm_installed),
                Span::raw(" ROCm Installed"),
            ]),
            Line::from(vec![
                status_icon(self.overview_data.pytorch_installed),
                Span::raw(" PyTorch Available"),
            ]),
            Line::from(vec![
                status_icon(self.overview_data.triton_installed),
                Span::raw(" Triton Ready"),
            ]),
            Line::from(vec![
                status_icon(self.overview_data.vllm_installed),
                Span::raw(" vLLM Ready"),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Tip: ", Style::default().fg(Color::Cyan)),
                Span::raw("Press 2 to install components"),
            ]),
        ];

        let status = Paragraph::new(Text::from(status_lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(" Status ", Style::default().fg(Color::Green)))
                    .border_style(Style::default().fg(Color::Green)),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(status, chunks[1]);
    }

    /// Draws the install tab.
    fn draw_install(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left: Component list
        let items: Vec<ListItem> = self.components.iter().enumerate().map(|(idx, comp)| {
            let indicator = if comp.selected { "[x]" } else { "[ ]" };
            let status = if comp.installed {
                Span::styled(" (installed)", Style::default().fg(Color::Green))
            } else {
                Span::raw("")
            };

            let style = if idx == self.selected_component {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", indicator), Style::default().fg(Color::Yellow)),
                Span::styled(&comp.name, style),
                status,
            ]))
        }).collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(" Components ", Style::default().fg(Color::Yellow)))
                    .border_style(Style::default().fg(Color::Yellow)),
            );
        frame.render_widget(list, chunks[0]);

        // Right: Component details
        let selected = &self.components[self.selected_component];
        let detail_lines = vec![
            Line::from(vec![
                Span::styled("Component: ", Style::default().fg(Color::Gray)),
                Span::styled(&selected.name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Description:", Style::default().fg(Color::Yellow)),
            ]),
            Line::from(selected.description.clone()),
            Line::from(""),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Gray)),
                if selected.installed {
                    Span::styled("Installed", Style::default().fg(Color::Green))
                } else {
                    Span::styled("Not installed", Style::default().fg(Color::Yellow))
                },
            ]),
            Line::from(vec![
                Span::styled("Selected: ", Style::default().fg(Color::Gray)),
                if selected.selected {
                    Span::styled("Yes", Style::default().fg(Color::Cyan))
                } else {
                    Span::styled("No", Style::default().fg(Color::DarkGray))
                },
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Controls:", Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("  ↑/↓ ", Style::default().fg(Color::Cyan)),
                Span::raw("Navigate"),
            ]),
            Line::from(vec![
                Span::styled("  Space ", Style::default().fg(Color::Cyan)),
                Span::raw("Toggle selection"),
            ]),
            Line::from(vec![
                Span::styled("  Enter ", Style::default().fg(Color::Cyan)),
                Span::raw("Start installation"),
            ]),
        ];

        let details = Paragraph::new(Text::from(detail_lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(" Details ", Style::default().fg(Color::Cyan)))
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(details, chunks[1]);
    }

    /// Draws the verify tab.
    fn draw_verify(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)])
            .split(area);

        let rows: Vec<Row> = self.verify_results.iter().map(|(name, passed, msg)| {
            let status_style = if *passed {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Yellow)
            };
            let status_text = if *passed { "✓ Passed" } else if self.verifying { "○ Running..." } else { "○ Pending" };

            Row::new(vec![
                Cell::from(name.clone()),
                Cell::from(Span::styled(status_text, status_style)),
                Cell::from(msg.clone()),
            ])
        }).collect();

        let header = Row::new(vec![
            Cell::from(Span::styled("Component", Style::default().add_modifier(Modifier::BOLD))),
            Cell::from(Span::styled("Status", Style::default().add_modifier(Modifier::BOLD))),
            Cell::from(Span::styled("Message", Style::default().add_modifier(Modifier::BOLD))),
        ]).style(Style::default().fg(Color::Cyan));

        let table = Table::new(
            rows,
            [
                Constraint::Percentage(25),
                Constraint::Percentage(20),
                Constraint::Percentage(55),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(" Verification Results ", Style::default().fg(Color::Green)))
                .border_style(Style::default().fg(Color::Green)),
        )
        .column_spacing(2);

        frame.render_widget(table, chunks[0]);

        // Hint line
        let hint = if self.verifying {
            "Running verification... Please wait..."
        } else {
            "Press V to run verification | Tab to switch tabs | Q to quit"
        };
        let hint_paragraph = Paragraph::new(Text::from(hint))
            .style(Style::default().fg(Color::Gray));
        frame.render_widget(hint_paragraph, chunks[1]);
    }

    /// Draws the settings tab.
    fn draw_settings(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left: Settings list
        let items: Vec<ListItem> = self.settings.iter().enumerate().map(|(idx, (name, value, _))| {
            let style = if idx == self.selected_setting {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::styled(name, style),
                Span::raw(": "),
                Span::styled(value, Style::default().fg(Color::Yellow)),
            ]))
        }).collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(" Settings ", Style::default().fg(Color::Magenta)))
                    .border_style(Style::default().fg(Color::Magenta)),
            );
        frame.render_widget(list, chunks[0]);

        // Right: Setting description
        let (name, value, desc) = &self.settings[self.selected_setting];
        let detail_lines = vec![
            Line::from(vec![
                Span::styled("Setting: ", Style::default().fg(Color::Gray)),
                Span::styled(name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Current Value: ", Style::default().fg(Color::Gray)),
                Span::styled(value, Style::default().fg(Color::Yellow)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Description:", Style::default().fg(Color::Yellow)),
            ]),
            Line::from(desc.clone()),
            Line::from(""),
            Line::from(vec![
                Span::styled("Controls:", Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("  ↑/↓ ", Style::default().fg(Color::Cyan)),
                Span::raw("Navigate"),
            ]),
            Line::from(vec![
                Span::styled("  Enter ", Style::default().fg(Color::Cyan)),
                Span::raw("Edit setting"),
            ]),
        ];

        let details = Paragraph::new(Text::from(detail_lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(" Details ", Style::default().fg(Color::Cyan)))
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(details, chunks[1]);
    }
}

impl Default for MainScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for MainScreen {
    fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        // Draw tabs with Rusty Stack styling
        let titles: Vec<_> = self.tabs.iter().enumerate().map(|(idx, t)| {
            if idx == self.current_tab {
                Span::styled(t.as_str(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            } else {
                Span::styled(t.as_str(), Style::default().fg(Color::Gray))
            }
        }).collect();

        let tabs = Tabs::new(titles)
            .select(self.current_tab)
            .block(
                Block::default()
                    .title(Span::styled(" Rusty Stack ", Style::default().fg(Color::Cyan)))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .divider(Span::styled(" | ", Style::default().fg(Color::DarkGray)));
        frame.render_widget(tabs, chunks[0]);

        // Draw content based on current tab
        match self.current_tab {
            0 => self.draw_overview(frame, chunks[1]),
            1 => self.draw_install(frame, chunks[1]),
            2 => self.draw_verify(frame, chunks[1]),
            3 => self.draw_settings(frame, chunks[1]),
            _ => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<()> {
        use crossterm::event::KeyCode;

        let mut changed = false;
        match key.code {
            KeyCode::Up => {
                match self.current_tab {
                    1 => {
                        if self.selected_component > 0 {
                            self.selected_component -= 1;
                            changed = true;
                        }
                    }
                    3 => {
                        if self.selected_setting > 0 {
                            self.selected_setting -= 1;
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
            KeyCode::Down => {
                match self.current_tab {
                    1 => {
                        if self.selected_component < self.components.len() - 1 {
                            self.selected_component += 1;
                            changed = true;
                        }
                    }
                    3 => {
                        if self.selected_setting < self.settings.len() - 1 {
                            self.selected_setting += 1;
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
            KeyCode::Char(' ') => {
                if self.current_tab == 1 {
                    self.components[self.selected_component].selected =
                        !self.components[self.selected_component].selected;
                    changed = true;
                }
            }
            KeyCode::Char('v') | KeyCode::Char('V') => {
                if self.current_tab == 2 {
                    // Trigger verification when on verify tab
                    self.start_verification();
                    changed = true;
                }
            }
            _ => {}
        }
        if changed {
            self.dirty = true;
        }
        Ok(())
    }

    fn update(&mut self) {
        // Update animation or data
        self.poll_verification();
    }

    fn next_tab(&mut self) {
        self.current_tab = (self.current_tab + 1) % self.tabs.len();
        self.dirty = true;
    }

    fn prev_tab(&mut self) {
        if self.current_tab == 0 {
            self.current_tab = self.tabs.len() - 1;
        } else {
            self.current_tab -= 1;
        }
        self.dirty = true;
    }

    fn set_tab(&mut self, tab: usize) {
        if tab < self.tabs.len() {
            self.current_tab = tab;
            self.dirty = true;
        }
    }

    fn is_dirty(&self) -> bool {
        self.dirty
    }

    fn mark_clean(&mut self) {
        self.dirty = false;
    }
}

/// Installation screen.
pub struct InstallScreen {
    /// Selected components
    selected: Vec<String>,
    /// Installation in progress
    installing: bool,
}

impl InstallScreen {
    /// Creates a new install screen.
    pub fn new() -> Self {
        Self {
            selected: Vec::new(),
            installing: false,
        }
    }

    /// Toggles component selection.
    pub fn toggle_component(&mut self, component: impl Into<String>) {
        let component = component.into();
        if let Some(pos) = self.selected.iter().position(|x| x == &component) {
            self.selected.remove(pos);
        } else {
            self.selected.push(component);
        }
    }
}

impl Default for InstallScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for InstallScreen {
    fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let block = Block::default()
            .title(Span::styled(" Installation ", Style::default().fg(Color::Green)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green));

        let text = if self.installing {
            Text::from("Installation in progress...")
        } else {
            Text::from(format!(
                "Select components to install:\n{}\n\nPress Space to toggle, Enter to install",
                if self.selected.is_empty() {
                    "None selected".to_string()
                } else {
                    self.selected.join(", ")
                }
            ))
        };

        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(paragraph, area);
    }

    fn handle_key(&mut self, _key: KeyEvent) -> Result<()> {
        Ok(())
    }

    fn update(&mut self) {}
    fn next_tab(&mut self) {}
    fn prev_tab(&mut self) {}
    fn set_tab(&mut self, _tab: usize) {}
}

/// Status/verification screen.
pub struct StatusScreen {
    /// Status items
    items: Vec<(String, bool)>,
}

impl StatusScreen {
    /// Creates a new status screen.
    pub fn new() -> Self {
        Self {
            items: vec![
                ("ROCm".to_string(), false),
                ("PyTorch".to_string(), false),
                ("Triton".to_string(), false),
            ],
        }
    }

    /// Updates status of an item.
    pub fn set_status(&mut self, item: impl Into<String>, status: bool) {
        let item = item.into();
        if let Some(pos) = self.items.iter().position(|(i, _)| i == &item) {
            self.items[pos].1 = status;
        }
    }
}

impl Default for StatusScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for StatusScreen {
    fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let block = Block::default()
            .title(Span::styled(" Status ", Style::default().fg(Color::Blue)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue));

        let lines: Vec<Line> = self
            .items
            .iter()
            .map(|(name, status)| {
                let status_str = if *status { "✓" } else { "✗" };
                let color = if *status { Color::Green } else { Color::Red };
                Line::from(vec![
                    Span::styled(status_str, Style::default().fg(color)),
                    Span::raw(" "),
                    Span::raw(name.clone()),
                ])
            })
            .collect();

        let text = Text::from(lines);
        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(paragraph, area);
    }

    fn handle_key(&mut self, _key: KeyEvent) -> Result<()> {
        Ok(())
    }

    fn update(&mut self) {}
    fn next_tab(&mut self) {}
    fn prev_tab(&mut self) {}
    fn set_tab(&mut self, _tab: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_screen_creation() {
        let screen = MainScreen::new();
        assert_eq!(screen.current_tab, 0);
        assert!(!screen.tabs.is_empty());
    }

    #[test]
    fn test_main_screen_tab_navigation() {
        let mut screen = MainScreen::new();
        let initial_tab = screen.current_tab;

        screen.next_tab();
        assert_ne!(screen.current_tab, initial_tab);

        screen.prev_tab();
        assert_eq!(screen.current_tab, initial_tab);
    }

    #[test]
    fn test_install_screen_component_toggle() {
        let mut screen = InstallScreen::new();
        screen.toggle_component("ROCm");
        assert!(screen.selected.contains(&"ROCm".to_string()));

        screen.toggle_component("ROCm");
        assert!(!screen.selected.contains(&"ROCm".to_string()));
    }

    #[test]
    fn test_status_screen() {
        let mut screen = StatusScreen::new();
        screen.set_status("ROCm", true);

        let rocm_item = screen.items.iter().find(|(n, _)| n == "ROCm");
        assert!(rocm_item.is_some());
        assert!(rocm_item.unwrap().1);
    }

    #[test]
    fn test_set_tab() {
        let mut screen = MainScreen::new();
        screen.set_tab(2);
        assert_eq!(screen.current_tab, 2);
    }
}
