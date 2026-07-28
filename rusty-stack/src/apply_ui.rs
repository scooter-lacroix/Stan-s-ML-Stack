//! Inline, animated apply-phase UI for `rusty-stack update`.
//!
//! Replaces the raw per-line `println!`/`eprint!` flood with a single
//! ratatui `Viewport::Inline` panel: a bordered box showing one row per
//! component (status glyph + name + `current → target` + progress bar +
//! latest milestone message), a footer tally, and per-row gradient spinners.
//! The panel redraws in place on every installer event; the full unfiltered
//! log still goes to the log file (stdout/stderr stay clean for the TTY).
//!
//! # Design
//!
//! - [`ApplyState`] and [`ApplyRow`] are plain data; [`ApplyState::apply_event`]
//!   is a pure reducer over [`rusty_stack::installer::InstallerEvent`]. These
//!   are unit-tested without a live terminal.
//! - [`render`] draws the state into a ratatui `Frame`. It only reads state, so
//!   the reducer ([`ApplyState::apply_event`]) and [`ApplyState::tally`] are
//!   unit-tested without a live terminal.
//! - Non-TTY / JSON callers never construct the panel (see [`ApplyPanel`]'s
//!   `None` fallback), so machine-readable output is unchanged.

use rusty_stack::installer::InstallerEvent;
use std::io::{IsTerminal, Stdout};

#[cfg(feature = "tui")]
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};

// ===========================================================================
// Animation primitives (pure, testable)
// ===========================================================================

/// Braille spinner frames — cycled by [`ApplyState::spinner_frame`] on each
/// redraw tick (the apply loop's 100 ms `recv_timeout` wake advances it), so the
/// glyph rotates smoothly even when the installer emits no events for a moment.
const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Spinner glyph for a given frame index.
pub fn spinner_char(frame: usize) -> char {
    SPINNER[frame % SPINNER.len()]
}

/// HSV → RGB (h in 0..=1, s/v in 0..=1). Pure, no deps.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let hh = (h * 6.0) % 6.0;
    let x = c * (1.0 - (hh % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match hh {
        q if q < 1.0 => (c, x, 0.0),
        q if q < 2.0 => (x, c, 0.0),
        q if q < 3.0 => (0.0, c, x),
        q if q < 4.0 => (0.0, x, c),
        q if q < 5.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    )
}

/// Gradient color for frame `frame`: full hue cycle at ~12°/frame (≈3 s per
/// revolution at 10 fps), fixed saturation/value for a vivid spinner + border.
/// Drives both the running-row glyph and the panel border so the whole panel
/// breathes as one animated surface.
#[cfg(feature = "tui")]
pub fn frame_color(frame: usize) -> Color {
    let hue = ((frame as f32 * 12.0) % 360.0) / 360.0;
    let (r, g, b) = hsv_to_rgb(hue, 0.65, 1.0);
    Color::Rgb(r, g, b)
}

// ===========================================================================
// State (pure, testable)
// ===========================================================================

/// Per-component lifecycle shown in the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowStatus {
    /// Not yet started.
    #[default]
    Pending,
    /// Install/verify in progress.
    Running,
    /// Completed + verified.
    Done,
    /// Failed (install or verification).
    Failed,
}

impl RowStatus {
    /// Stable glyph for the row (rendered with color by the frontend).
    pub fn glyph(self) -> &'static str {
        match self {
            RowStatus::Pending => " ",
            RowStatus::Running => "⠋",
            RowStatus::Done => "✓",
            RowStatus::Failed => "✗",
        }
    }

    /// ratatui color for the glyph.
    #[cfg(feature = "tui")]
    pub fn color(self) -> Color {
        match self {
            RowStatus::Pending => Color::DarkGray,
            RowStatus::Running => Color::Yellow,
            RowStatus::Done => Color::Green,
            RowStatus::Failed => Color::Red,
        }
    }
}

/// One component row in the apply panel.
#[derive(Debug, Clone)]
pub struct ApplyRow {
    pub id: String,
    pub name: String,
    pub current: String,
    pub target: String,
    pub status: RowStatus,
    /// 0.0..=1.0 — fraction complete reported by the installer.
    pub progress: f32,
    /// Latest milestone message (filtered; raw logs go to the file only).
    pub message: String,
}

impl ApplyRow {
    /// Build a pending row from plan data.
    pub fn new(id: &str, name: &str, current: &str, target: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            current: current.to_string(),
            target: target.to_string(),
            status: RowStatus::Pending,
            progress: 0.0,
            message: String::new(),
        }
    }
}

/// Full apply-panel state: the ordered component rows + a rolling log ticker.
#[derive(Debug, Clone, Default)]
pub struct ApplyState {
    pub rows: Vec<ApplyRow>,
    /// Recent filtered log lines shown in a footer ticker (newest last).
    pub ticker: Vec<String>,
    pub title: String,
    /// Animation frame — advanced once per redraw tick (≈100 ms) so the braille
    /// spinner rotates and the border hue cycles independent of installer events.
    pub spinner_frame: usize,
}

const TICKER_LEN: usize = 2;

impl ApplyState {
    /// Initialize from the plan's component ids/names/versions.
    pub fn from_plan<I, S>(title: &str, items: I) -> Self
    where
        I: IntoIterator<Item = (S, S, S, S)>,
        S: AsRef<str>,
    {
        let rows = items
            .into_iter()
            .map(|(id, name, current, target)| {
                ApplyRow::new(
                    id.as_ref(),
                    name.as_ref(),
                    current.as_ref(),
                    target.as_ref(),
                )
            })
            .collect();
        Self {
            rows,
            ticker: Vec::new(),
            title: title.to_string(),
            spinner_frame: 0,
        }
    }

    /// Pure reducer: fold one installer event into the state.
    ///
    /// `running_id` is the component currently being applied (the only one that
    /// can receive `Progress`/`Log` events), so `Log`/`Progress` events target
    /// it without needing to match by id on every line.
    pub fn apply_event(&mut self, ev: &InstallerEvent, running_id: &str) {
        match ev {
            InstallerEvent::ComponentStart { component_id, name } => {
                if let Some(row) = self.rows.iter_mut().find(|r| r.id == *component_id) {
                    row.status = RowStatus::Running;
                    row.progress = 0.0;
                    // Always take the friendly name from the event (pending rows
                    // show the id; started rows show e.g. "ONNX Runtime").
                    row.name = name.clone();
                    row.message = format!("Installing {}…", name);
                }
            }
            InstallerEvent::Progress {
                progress, message, ..
            } => {
                if let Some(row) = self.rows.iter_mut().find(|r| r.id == running_id) {
                    row.progress = progress.clamp(0.0, 1.0);
                    if !message.is_empty() {
                        row.message = message.clone();
                    }
                    row.status = RowStatus::Running;
                }
            }
            InstallerEvent::Log(line, important) => {
                let milestone = is_milestone(line);
                if milestone {
                    if let Some(row) = self.rows.iter_mut().find(|r| r.id == running_id) {
                        // Milestones (e.g. "Successfully installed …") replace the
                        // running message; non-milestone pip chatter is ignored on
                        // the TTY (still written to the log file by the caller).
                        row.message = line.clone();
                    }
                }
                // Surface milestones AND explicitly-important (error) lines —
                // once, never twice for a line that is both.
                if milestone || *important {
                    self.push_ticker(line.clone());
                }
            }
            InstallerEvent::ComponentComplete {
                component_id,
                success,
                message,
            } => {
                if let Some(row) = self.rows.iter_mut().find(|r| r.id == *component_id) {
                    row.status = if *success {
                        RowStatus::Done
                    } else {
                        RowStatus::Failed
                    };
                    row.progress = if *success { 1.0 } else { row.progress };
                    row.message = message.clone();
                }
            }
            InstallerEvent::VerificationReport {
                component_id,
                lines,
            } => {
                if let Some(row) = self.rows.iter_mut().find(|r| r.id == *component_id) {
                    // Keep the one-line verification outcome; skip the full dump
                    // (it's in the log file).
                    if let Some(last) = lines.last() {
                        row.message = last.clone();
                    }
                }
            }
            InstallerEvent::Finished { .. } => {}
        }
    }

    fn push_ticker(&mut self, line: String) {
        let clean = line.trim().to_string();
        if clean.is_empty() {
            return;
        }
        self.ticker.push(clean);
        if self.ticker.len() > TICKER_LEN {
            self.ticker.drain(0..self.ticker.len() - TICKER_LEN);
        }
    }

    /// Tally (done, failed, in_flight, pending).
    pub fn tally(&self) -> (usize, usize, usize, usize) {
        let mut done = 0;
        let mut failed = 0;
        let mut running = 0;
        let mut pending = 0;
        for r in &self.rows {
            match r.status {
                RowStatus::Done => done += 1,
                RowStatus::Failed => failed += 1,
                RowStatus::Running => running += 1,
                RowStatus::Pending => pending += 1,
            }
        }
        (done, failed, running, pending)
    }
}

/// A line worth showing on the TTY (vs. pip chatter that goes only to the log).
fn is_milestone(line: &str) -> bool {
    let l = line.to_lowercase();
    l.contains("successfully installed")
        || l.contains("already installed")
        || l.contains("installed ")
        || l.contains("error")
        || l.contains("fail")
        || l.contains("verif")
        || l.contains("pulling")
        || l.contains("cloning")
        || l.contains("version")
        || l.contains("%)")
}

/// Compact ASCII progress bar (10 cells). The TTY uses a ratatui `Gauge`; this
/// is for the pure text fallback + tests.
fn progress_bar(p: f32) -> String {
    let filled = ((p.clamp(0.0, 1.0) * 10.0).round() as usize).min(10);
    let mut s = String::with_capacity(10);
    for i in 0..10 {
        s.push(if i < filled { '█' } else { '░' });
    }
    s
}

/// Truncate `s` to fit `width` display columns, appending an ellipsis if cut.
/// Uses char count as an approximation of display width (error messages are
/// overwhelmingly ASCII) so the panel needs no extra unicode-width dependency.
/// Guarantees the returned string never exceeds `width` chars, keeping long
/// errors/milestones INSIDE the bordered panel instead of overflowing it.
fn truncate_to_width(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if s.chars().count() <= width {
        return s.to_string();
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

// ===========================================================================
// ratatui rendering (tui feature only)
// ===========================================================================

#[cfg(feature = "tui")]
/// Draw the apply state into a ratatui frame.
pub fn render(frame: &mut Frame<'_>, area: Rect, state: &ApplyState) {
    let chunks = Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).split(area);
    let panel = chunks[0];
    let footer = chunks[1];

    let name_w = state
        .rows
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(8)
        .max(8);

    let mut lines: Vec<Line> = Vec::with_capacity(state.rows.len() + 1);
    lines.push(Line::styled(
        format!(
            "  {:<name_w$}  {:>14}    {:>14}",
            "COMPONENT",
            "CURRENT",
            "TARGET",
            name_w = name_w,
        ),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray),
    ));

    for r in &state.rows {
        let pct = ((r.progress * 100.0) as u16).min(100);
        // Running rows animate: the braille glyph rotates frame-to-frame and
        // takes the gradient color, so the eye tracks the active component even
        // when no Progress event has landed for a moment.
        let glyph_text = if r.status == RowStatus::Running {
            spinner_char(state.spinner_frame).to_string()
        } else {
            r.status.glyph().to_string()
        };
        let glyph_color = if r.status == RowStatus::Running {
            frame_color(state.spinner_frame)
        } else {
            r.status.color()
        };
        let glyph = Span::styled(
            format!("{} ", glyph_text),
            Style::default()
                .fg(glyph_color)
                .add_modifier(Modifier::BOLD),
        );
        let name = Span::styled(
            format!("{:<name_w$}  ", r.name, name_w = name_w),
            Style::default().fg(if r.status == RowStatus::Running {
                Color::Yellow
            } else {
                Color::White
            }),
        );
        let cur = Span::raw(format!(
            "{:>14} ",
            if r.current.is_empty() {
                "—"
            } else {
                &r.current
            }
        ));
        let arrow = Span::styled("→ ", Style::default().fg(Color::DarkGray));
        let tgt = Span::raw(format!(
            "{:>14} ",
            if r.target.is_empty() {
                "—"
            } else {
                &r.target
            }
        ));
        // Running rows show a live bar + pct; completed/failed rows show their
        // outcome message; idle rows show the pct.
        let msg = if r.status == RowStatus::Running {
            format!("{} {:>3}%", progress_bar(r.progress), pct)
        } else if r.message.is_empty() {
            format!("{:>3}%", pct)
        } else {
            r.message.clone()
        };
        // Keep the message inside the panel: available columns = content width
        // (panel minus 2 borders) minus the fixed glyph/name/current/arrow/target
        // columns (name_w + 36). Long errors no longer break the bounding box.
        let msg_w = (panel.width as usize).saturating_sub(name_w + 38).max(8);
        let msg = truncate_to_width(&msg, msg_w);
        let msg_span = Span::styled(
            msg,
            Style::default().fg(if r.status == RowStatus::Failed {
                Color::Red
            } else {
                Color::DarkGray
            }),
        );
        lines.push(Line::from(vec![glyph, name, cur, arrow, tgt, msg_span]));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {} ", state.title))
        .title_alignment(Alignment::Left)
        // Animated border: hue cycles with the spinner frame so the panel
        // border breathes in lockstep with the active-row glyph.
        .border_style(Style::default().fg(frame_color(state.spinner_frame)));
    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, panel);

    let (done, failed, running, pending) = state.tally();
    let tally_line = Line::from(vec![
        Span::styled(format!("  {done} ok",), Style::default().fg(Color::Green)),
        Span::raw(" · "),
        Span::styled(format!("{failed} failed"), Style::default().fg(Color::Red)),
        Span::raw(" · "),
        Span::styled(
            format!("{running} in flight"),
            Style::default().fg(Color::Yellow),
        ),
        Span::raw(" · "),
        Span::raw(format!("{pending} pending")),
        Span::raw("   [logs: ~/.mlstack/logs]"),
    ]);
    frame.render_widget(
        Paragraph::new(tally_line).alignment(Alignment::Left),
        footer,
    );
}

// ===========================================================================
// Live panel (terminal lifecycle)
// ===========================================================================

#[cfg(feature = "tui")]
/// Owns the inline ratatui terminal for the apply phase. `None` (via
/// [`ApplyPanel::disabled`]) means non-TTY/JSON — callers keep legacy output.
pub struct ApplyPanel {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    state: ApplyState,
    enabled: bool,
}

#[cfg(feature = "tui")]
impl ApplyPanel {
    /// Create the inline panel. `rows` = (id, name, current, target) per item.
    /// Returns a disabled panel if the terminal isn't a TTY (caller keeps
    /// legacy stdout output in that case).
    pub fn try_new(title: &str, rows: Vec<(&str, &str, &str, &str)>) -> Option<Self> {
        if !std::io::stdout().is_terminal() {
            return None;
        }
        let height = (rows.len() as u16 + 4).min(40); // rows + header + tally + borders
        let backend = CrosstermBackend::new(std::io::stdout());
        let terminal = Terminal::with_options(
            backend,
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Inline(height),
            },
        )
        .ok()?;
        crossterm::terminal::enable_raw_mode().ok()?;
        let state = ApplyState::from_plan(title, rows.iter().map(|(a, b, c, d)| (*a, *b, *c, *d)));
        Some(Self {
            terminal,
            state,
            enabled: true,
        })
    }

    /// Access the mutable state (apply events fold into it).
    pub fn state_mut(&mut self) -> &mut ApplyState {
        &mut self.state
    }

    /// Redraw the panel from current state.
    pub fn draw(&mut self) {
        if !self.enabled {
            return;
        }
        let _ = self.terminal.draw(|f| {
            let area = f.area();
            render(f, area, &self.state);
        });
    }

    /// Advance the animation frame by one and redraw. Called by the apply loop
    /// on each `recv_timeout` wake (≈100 ms) so the spinner rotates and the
    /// border hue cycles even when the installer is momentarily silent.
    pub fn tick(&mut self) {
        if !self.enabled {
            return;
        }
        self.state.spinner_frame = self.state.spinner_frame.wrapping_add(1);
        self.draw();
    }
}

#[cfg(feature = "tui")]
impl Drop for ApplyPanel {
    fn drop(&mut self) {
        if self.enabled {
            // Best-effort final frame, then restore the terminal (raw mode off,
            // cursor shown). Drop runs even on panic/early return, so the panel
            // never leaves the terminal in raw mode.
            let _ = self.terminal.draw(|f| {
                render(f, f.area(), &self.state);
            });
            let _ = crossterm::terminal::disable_raw_mode();
            let _ = crossterm::execute!(std::io::stdout(), crossterm::cursor::Show);
            self.enabled = false;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ev_progress(progress: f32, msg: &str) -> InstallerEvent {
        InstallerEvent::Progress {
            component_id: "onnx".into(),
            progress,
            message: msg.into(),
        }
    }

    #[test]
    fn row_status_glyphs() {
        assert_eq!(RowStatus::Pending.glyph(), " ");
        assert_eq!(RowStatus::Running.glyph(), "⠋");
        assert_eq!(RowStatus::Done.glyph(), "✓");
        assert_eq!(RowStatus::Failed.glyph(), "✗");
    }

    #[test]
    fn state_starts_all_pending() {
        let s = ApplyState::from_plan(
            "Update",
            [
                ("onnx", "ONNX", "1.23", "1.25"),
                ("aiter", "AITER", "0.0", "0.1"),
            ],
        );
        assert_eq!(s.rows.len(), 2);
        assert!(s.rows.iter().all(|r| r.status == RowStatus::Pending));
        assert_eq!(s.tally(), (0, 0, 0, 2));
    }

    #[test]
    fn start_then_progress_marks_running() {
        let mut s = ApplyState::from_plan("Update", [("onnx", "ONNX", "1.23", "1.25")]);
        s.apply_event(
            &InstallerEvent::ComponentStart {
                component_id: "onnx".into(),
                name: "ONNX Runtime".into(),
            },
            "onnx",
        );
        assert_eq!(s.rows[0].status, RowStatus::Running);
        s.apply_event(&ev_progress(0.5, "downloading"), "onnx");
        assert_eq!(s.rows[0].progress, 0.5);
        assert_eq!(s.rows[0].message, "downloading");
    }

    #[test]
    fn complete_marks_done_or_failed_and_clamps_progress() {
        let mut s = ApplyState::from_plan("U", [("onnx", "ONNX", "1.23", "1.25")]);
        s.apply_event(&ev_progress(2.0, "x"), "onnx"); // clamp >1
        assert_eq!(s.rows[0].progress, 1.0);
        s.apply_event(
            &InstallerEvent::ComponentComplete {
                component_id: "onnx".into(),
                success: true,
                message: "ok".into(),
            },
            "onnx",
        );
        assert_eq!(s.rows[0].status, RowStatus::Done);
        assert_eq!(s.tally(), (1, 0, 0, 0));

        let mut s2 = ApplyState::from_plan("U", [("onnx", "ONNX", "1.23", "1.25")]);
        s2.apply_event(
            &InstallerEvent::ComponentComplete {
                component_id: "onnx".into(),
                success: false,
                message: "boom".into(),
            },
            "onnx",
        );
        assert_eq!(s2.rows[0].status, RowStatus::Failed);
        assert_eq!(s2.tally(), (0, 1, 0, 0));
    }

    #[test]
    fn milestone_log_updates_message_pip_chatter_does_not() {
        let mut s = ApplyState::from_plan("U", [("onnx", "ONNX", "1.23", "1.25")]);
        s.apply_event(
            &InstallerEvent::ComponentStart {
                component_id: "onnx".into(),
                name: "ONNX Runtime".into(),
            },
            "onnx",
        );
        // pip chatter — not a milestone, must not overwrite the running message
        s.apply_event(
            &InstallerEvent::Log("Collecting numpy<2".into(), false),
            "onnx",
        );
        assert_eq!(s.rows[0].message, "Installing ONNX Runtime…");
        assert!(s.ticker.is_empty());

        // milestone — overwrites + lands in the ticker
        s.apply_event(
            &InstallerEvent::Log(
                "Successfully installed onnxruntime-migraphx-1.25.0".into(),
                false,
            ),
            "onnx",
        );
        assert_eq!(
            s.rows[0].message,
            "Successfully installed onnxruntime-migraphx-1.25.0"
        );
        assert_eq!(s.ticker.len(), 1);
    }

    #[test]
    fn important_log_always_tickered() {
        let mut s = ApplyState::from_plan("U", [("onnx", "ONNX", "1.23", "1.25")]);
        s.apply_event(
            &InstallerEvent::Log("[ERROR] ONNX Runtime failed: 404".into(), true),
            "onnx",
        );
        assert_eq!(s.ticker.len(), 1);
        assert!(s.ticker[0].contains("404"));
    }

    #[test]
    fn ticker_caps_at_two_lines() {
        let mut s = ApplyState::from_plan("U", [("onnx", "ONNX", "1.23", "1.25")]);
        for i in 0..5 {
            s.apply_event(
                &InstallerEvent::Log(format!("Successfully installed pkg-{i}"), true),
                "onnx",
            );
        }
        assert_eq!(s.ticker.len(), 2);
        assert_eq!(s.ticker[1], "Successfully installed pkg-4");
    }

    #[test]
    fn progress_bar_shapes() {
        assert_eq!(progress_bar(0.0), "░░░░░░░░░░");
        assert_eq!(progress_bar(1.0), "██████████");
        assert_eq!(progress_bar(0.5), "█████░░░░░");
    }

    #[test]
    fn truncate_to_width_fits_and_cuts() {
        // Short strings pass through unchanged.
        assert_eq!(truncate_to_width("ok", 10), "ok");
        // Exactly-width string is unchanged (no ellipsis).
        assert_eq!(truncate_to_width("abcd", 4), "abcd");
        // Over-width string is cut to width with an ellipsis.
        let long = "PyTorch with ROCm is a sealed CORE component — force ignored";
        let t = truncate_to_width(long, 20);
        assert_eq!(t.chars().count(), 20);
        assert!(t.ends_with('…'));
        assert!(t.starts_with("PyTorch with ROCm"));
        // width 0 → empty (never panics).
        assert_eq!(truncate_to_width("x", 0), "");
    }

    #[test]
    fn is_milestone_filters_pip_chatter() {
        assert!(is_milestone(
            "Successfully installed onnxruntime-migraphx-1.25.0"
        ));
        assert!(is_milestone("ERROR: Could not install"));
        assert!(!is_milestone("Collecting numpy"));
        assert!(!is_milestone("  Downloading numpy-2.5.1.whl"));
    }

    #[test]
    fn spinner_char_cycles_through_frames() {
        // Wraps modulo the braille sequence length, so consecutive frames differ
        // (the whole point of an animation) but frame == frame+len is identical.
        let len = SPINNER.len();
        assert_ne!(spinner_char(0), spinner_char(1));
        assert_ne!(spinner_char(1), spinner_char(2));
        assert_eq!(spinner_char(0), spinner_char(len));
        assert_eq!(spinner_char(7), spinner_char(7 + 3 * len));
    }

    #[test]
    fn hsv_to_rgb_stays_in_byte_range_and_extremes() {
        // Red (h=0), green (h=1/3), blue (h=2/3), and every frame of a full
        // hue cycle must map into 0..=255 — a clamping/format bug would overflow.
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!((r, g, b), (255, 0, 0));
        let (r, g, b) = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert_eq!((r, g, b), (0, 255, 0));
        let (r, g, b) = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0);
        assert_eq!((r, g, b), (0, 0, 255));
        // A gradient that returns the same color for every frame is no gradient
        // at all — the cycle must produce several distinct colors.
        let distinct: std::collections::HashSet<(u8, u8, u8)> = (0..30)
            .map(|f| {
                let h = ((f as f32 * 12.0) % 360.0) / 360.0;
                hsv_to_rgb(h, 0.65, 1.0)
            })
            .collect();
        assert!(distinct.len() > 5, "gradient produced too few colors");
    }
}
