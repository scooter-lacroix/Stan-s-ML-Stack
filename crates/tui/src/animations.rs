//! Animation System
//!
//! Progress and spinner animations for the TUI.

/// Animation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationState {
    /// Animation is running
    Running,
    /// Animation is paused
    Paused,
    /// Animation is stopped
    Stopped,
}

/// Base animation trait.
pub trait Animation {
    /// Advances the animation by one frame.
    fn tick(&mut self);
    /// Gets the current frame.
    fn current(&self) -> String;
    /// Gets the animation state.
    fn state(&self) -> AnimationState;
    /// Starts the animation.
    fn start(&mut self);
    /// Pauses the animation.
    fn pause(&mut self);
    /// Stops the animation.
    fn stop(&mut self);
}

/// Spinner animation with multiple styles.
pub struct SpinnerAnimation {
    /// Animation frames
    frames: Vec<&'static str>,
    /// Current frame index
    frame: usize,
    /// Animation state
    state: AnimationState,
    /// Style name
    style: SpinnerStyle,
}

/// Spinner visual styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinnerStyle {
    /// Dots style: ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
    Dots,
    /// Line style: \|/-
    Line,
    /// Block style: ▖▘▝▗
    Block,
}

impl SpinnerAnimation {
    /// Creates a new spinner animation.
    pub fn new(style: SpinnerStyle) -> Self {
        let frames = match style {
            SpinnerStyle::Dots => vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            SpinnerStyle::Line => vec!["|", "/", "-", "\\"],
            SpinnerStyle::Block => vec!["▖", "▘", "▝", "▗"],
        };

        Self {
            frames,
            frame: 0,
            state: AnimationState::Stopped,
            style,
        }
    }

    /// Creates a default dots spinner.
    pub fn default_dots() -> Self {
        Self::new(SpinnerStyle::Dots)
    }

    /// Gets the spinner style.
    pub fn style(&self) -> SpinnerStyle {
        self.style
    }
}

impl Animation for SpinnerAnimation {
    fn tick(&mut self) {
        if self.state == AnimationState::Running {
            self.frame = (self.frame + 1) % self.frames.len();
        }
    }

    fn current(&self) -> String {
        self.frames[self.frame].to_string()
    }

    fn state(&self) -> AnimationState {
        self.state
    }

    fn start(&mut self) {
        self.state = AnimationState::Running;
    }

    fn pause(&mut self) {
        self.state = AnimationState::Paused;
    }

    fn stop(&mut self) {
        self.state = AnimationState::Stopped;
        self.frame = 0;
    }
}

/// Progress bar animation.
pub struct ProgressAnimation {
    /// Current progress (0-100)
    progress: f64,
    /// Target progress
    target: f64,
    /// Animation speed
    speed: f64,
    /// Animation state
    state: AnimationState,
}

impl ProgressAnimation {
    /// Creates a new progress animation.
    pub fn new() -> Self {
        Self {
            progress: 0.0,
            target: 0.0,
            speed: 1.0,
            state: AnimationState::Stopped,
        }
    }

    /// Sets the target progress.
    pub fn set_target(&mut self, target: f64) {
        self.target = target.clamp(0.0, 100.0);
        self.state = AnimationState::Running;
    }

    /// Sets the animation speed.
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed;
    }

    /// Gets the current progress.
    pub fn progress(&self) -> f64 {
        self.progress
    }

    /// Checks if animation is complete.
    pub fn is_complete(&self) -> bool {
        (self.progress - self.target).abs() < 0.01
    }
}

impl Default for ProgressAnimation {
    fn default() -> Self {
        Self::new()
    }
}

impl Animation for ProgressAnimation {
    fn tick(&mut self) {
        if self.state == AnimationState::Running {
            let diff = self.target - self.progress;
            if diff.abs() > 0.01 {
                self.progress += diff * self.speed * 0.1;
            } else {
                self.progress = self.target;
                if self.progress >= 100.0 {
                    self.state = AnimationState::Stopped;
                }
            }
        }
    }

    fn current(&self) -> String {
        format!("{:.1}%", self.progress)
    }

    fn state(&self) -> AnimationState {
        self.state
    }

    fn start(&mut self) {
        self.state = AnimationState::Running;
    }

    fn pause(&mut self) {
        self.state = AnimationState::Paused;
    }

    fn stop(&mut self) {
        self.state = AnimationState::Stopped;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spinner_animation() {
        let mut spinner = SpinnerAnimation::default_dots();
        spinner.start();

        let first = spinner.current();
        spinner.tick();
        let second = spinner.current();

        assert_ne!(first, second);
        assert_eq!(spinner.state(), AnimationState::Running);
    }

    #[test]
    fn test_spinner_styles() {
        let dots = SpinnerAnimation::new(SpinnerStyle::Dots);
        let line = SpinnerAnimation::new(SpinnerStyle::Line);
        let block = SpinnerAnimation::new(SpinnerStyle::Block);

        assert_eq!(dots.style, SpinnerStyle::Dots);
        assert_eq!(line.style, SpinnerStyle::Line);
        assert_eq!(block.style, SpinnerStyle::Block);
    }

    #[test]
    fn test_progress_animation() {
        let mut progress = ProgressAnimation::new();
        progress.set_target(50.0);
        progress.start();

        // Tick multiple times to approach target
        for _ in 0..100 {
            progress.tick();
        }

        assert!((progress.progress() - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_progress_complete() {
        let mut progress = ProgressAnimation::new();
        progress.set_target(100.0);
        progress.set_speed(10.0); // Fast speed
        progress.start();

        // Tick until complete
        for _ in 0..200 {
            progress.tick();
        }

        assert!(progress.is_complete() || progress.progress() >= 99.0);
    }

    #[test]
    fn test_animation_state_transitions() {
        let mut spinner = SpinnerAnimation::default_dots();

        assert_eq!(spinner.state(), AnimationState::Stopped);

        spinner.start();
        assert_eq!(spinner.state(), AnimationState::Running);

        spinner.pause();
        assert_eq!(spinner.state(), AnimationState::Paused);

        spinner.stop();
        assert_eq!(spinner.state(), AnimationState::Stopped);
    }
}
