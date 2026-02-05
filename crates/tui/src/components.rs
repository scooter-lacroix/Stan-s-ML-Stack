//! UI Components
//!
//! Reusable UI components for the TUI with Rusty Stack branding.

use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
};

/// Progress bar component with ROCm branding.
pub struct ProgressBar {
    /// Current progress (0-100)
    pub progress: f64,
    /// Label to display
    pub label: String,
    /// Color of the progress bar
    pub color: Color,
}

impl ProgressBar {
    /// Creates a new progress bar.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            progress: 0.0,
            label: label.into(),
            color: Color::Cyan,
        }
    }

    /// Sets the progress.
    pub fn set_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 100.0);
    }

    /// Draws the progress bar.
    pub fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let gauge = Gauge::default()
            .block(
                Block::default()
                    .title(Span::styled(
                        self.label.clone(),
                        Style::default().fg(Color::Cyan),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .gauge_style(Style::default().fg(self.color).bg(Color::Black))
            .percent(self.progress as u16);
        frame.render_widget(gauge, area);
    }
}

/// Animated spinner component.
pub struct Spinner {
    /// Current frame index
    frame: usize,
    /// Animation frames
    frames: Vec<&'static str>,
    /// Message to display
    pub message: String,
}

impl Spinner {
    /// Creates a new spinner.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            frame: 0,
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            message: message.into(),
        }
    }

    /// Advances the animation.
    pub fn tick(&mut self) {
        self.frame = (self.frame + 1) % self.frames.len();
    }

    /// Gets the current frame.
    pub fn current(&self) -> &str {
        self.frames[self.frame]
    }

    /// Draws the spinner.
    pub fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let text = format!("{} {}", self.current(), self.message);
        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(Color::Magenta))
            .alignment(Alignment::Left);
        frame.render_widget(paragraph, area);
    }
}

/// GPU information panel with Rusty Stack styling.
pub struct GPUPanel {
    /// GPU information lines
    gpu_info: Vec<String>,
    /// Whether data is loading
    loading: bool,
    /// Spinner for loading state
    spinner_frame: usize,
}

impl GPUPanel {
    /// Creates a new GPU panel.
    pub fn new() -> Self {
        Self {
            gpu_info: vec!["Detecting GPU...".to_string()],
            loading: true,
            spinner_frame: 0,
        }
    }

    /// Updates GPU information.
    pub fn update_gpu_info(&mut self, info: Vec<String>) {
        self.gpu_info = info;
        self.loading = false;
    }

    /// Updates the panel (called every tick).
    pub fn update(&mut self) {
        if self.loading {
            self.spinner_frame = (self.spinner_frame + 1) % 10;
        }
    }

    /// Gets spinner frame.
    fn spinner(&self) -> &'static str {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        FRAMES[self.spinner_frame % FRAMES.len()]
    }

    /// Draws the GPU panel.
    pub fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let block = Block::default()
            .title(Span::styled(
                " GPU Information ",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let lines: Vec<Line> = if self.loading {
            vec![
                Line::from(vec![
                    Span::styled(self.spinner(), Style::default().fg(Color::Magenta)),
                    Span::raw(" "),
                    Span::styled("Detecting hardware...", Style::default().fg(Color::Yellow)),
                ]),
            ]
        } else {
            self.gpu_info
                .iter()
                .map(|info| {
                    // Parse key-value pairs for styling
                    if let Some((key, value)) = info.split_once(':') {
                        Line::from(vec![
                            Span::styled(
                                format!("{}:", key),
                                Style::default().fg(Color::Gray),
                            ),
                            Span::styled(
                                value.to_string(),
                                Style::default().fg(Color::White),
                            ),
                        ])
                    } else {
                        Line::from(Span::styled(info.clone(), Style::default().fg(Color::White)))
                    }
                })
                .collect()
        };

        let text = Text::from(lines);
        let paragraph = Paragraph::new(text).block(block).wrap(Wrap { trim: true });
        frame.render_widget(paragraph, area);
    }
}

impl Default for GPUPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Log message panel with scrollback and Rusty Stack styling.
pub struct LogPanel {
    /// Log messages
    messages: Vec<String>,
    /// Maximum number of messages to keep
    max_messages: usize,
    /// Scroll offset
    scroll: usize,
}

impl LogPanel {
    /// Creates a new log panel.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: 1000,
            scroll: 0,
        }
    }

    /// Adds a message to the log.
    pub fn add_message(&mut self, message: String) {
        self.messages.push(message);
        if self.messages.len() > self.max_messages {
            self.messages.remove(0);
        }
        // Auto-scroll to bottom
        self.scroll = self.messages.len().saturating_sub(1);
    }

    /// Scrolls up.
    pub fn scroll_up(&mut self) {
        self.scroll = self.scroll.saturating_sub(1);
    }

    /// Scrolls down.
    pub fn scroll_down(&mut self) {
        if self.scroll < self.messages.len().saturating_sub(1) {
            self.scroll += 1;
        }
    }

    /// Updates the panel.
    pub fn update(&mut self) {
        // Could process new log messages here
    }

    /// Draws the log panel.
    pub fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let block = Block::default()
            .title(Span::styled(
                " Installation Log ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        let visible_height = area.height.saturating_sub(2) as usize;
        let start = self.messages.len().saturating_sub(visible_height);

        let visible_messages: Vec<ListItem> = if self.messages.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No log messages yet",
                Style::default().fg(Color::DarkGray),
            )))]
        } else {
            self.messages
                .iter()
                .skip(start)
                .take(visible_height)
                .map(|m| {
                    // Color log messages based on content
                    let style = if m.contains("Error") || m.contains("error") || m.contains("failed") {
                        Style::default().fg(Color::Red)
                    } else if m.contains("Warning") || m.contains("warning") {
                        Style::default().fg(Color::Yellow)
                    } else if m.contains("complete") || m.contains("success") || m.contains("✓") {
                        Style::default().fg(Color::Green)
                    } else if m.contains("Starting") || m.contains("Detecting") {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default().fg(Color::Gray)
                    };
                    ListItem::new(Line::from(Span::styled(m.as_str(), style)))
                })
                .collect()
        };

        let list = List::new(visible_messages).block(block);
        frame.render_widget(list, area);
    }
}

impl Default for LogPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Status bar component with Rusty Stack styling.
pub struct StatusBar {
    /// Current status message
    pub status: String,
    /// Key hints
    pub hints: Vec<String>,
}

impl StatusBar {
    /// Creates a new status bar.
    pub fn new() -> Self {
        Self {
            status: "Ready".to_string(),
            hints: vec![
                "Q:Quit".to_string(),
                "Tab:Switch".to_string(),
                "1-4:Tabs".to_string(),
                "R:Refresh".to_string(),
                "↑↓:Navigate".to_string(),
            ],
        }
    }

    /// Sets the status message.
    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    /// Draws the status bar.
    pub fn draw(&self, frame: &mut ratatui::Frame, area: Rect) {
        let hints_spans: Vec<Span> = self.hints.iter().enumerate().flat_map(|(i, hint)| {
            let mut spans = vec![
                Span::styled(hint, Style::default().fg(Color::Cyan)),
            ];
            if i < self.hints.len() - 1 {
                spans.push(Span::styled(" | ", Style::default().fg(Color::DarkGray)));
            }
            spans
        }).collect();

        let mut all_spans = vec![
            Span::styled("✦ ", Style::default().fg(Color::Yellow)),
            Span::styled(&self.status, Style::default().fg(Color::White)),
            Span::styled(" | ", Style::default().fg(Color::DarkGray)),
        ];
        all_spans.extend(hints_spans);

        let paragraph = Paragraph::new(Line::from(all_spans))
            .style(Style::default().bg(Color::Black))
            .alignment(Alignment::Left);
        frame.render_widget(paragraph, area);
    }
}

impl Default for StatusBar {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar() {
        let mut bar = ProgressBar::new("Test");
        bar.set_progress(50.0);
        assert_eq!(bar.progress, 50.0);
    }

    #[test]
    fn test_spinner() {
        let mut spinner = Spinner::new("Loading");
        let first = spinner.current().to_string();
        spinner.tick();
        let second = spinner.current().to_string();
        assert_ne!(first, second);
    }

    #[test]
    fn test_log_panel() {
        let mut panel = LogPanel::new();
        panel.add_message("Test message".to_string());
        assert_eq!(panel.messages.len(), 1);
    }

    #[test]
    fn test_status_bar() {
        let mut bar = StatusBar::new();
        bar.set_status("Installing");
        assert_eq!(bar.status, "Installing");
    }

    #[test]
    fn test_gpu_panel_update() {
        let mut panel = GPUPanel::new();
        assert!(panel.loading);
        panel.update_gpu_info(vec!["GPU: AMD Radeon".to_string()]);
        assert!(!panel.loading);
    }
}
