//! ML Stack TUI
//!
//! Terminal UI for Stan's ML Stack with animations, progress indicators,
//! and Rusty Stack branding.

pub mod animations;
pub mod app;
pub mod components;
pub mod events;
pub mod screens;

pub use animations::{Animation, AnimationState, ProgressAnimation, SpinnerAnimation};
pub use app::{App, AppState, HardwareInfo, Stage};
pub use components::{GPUPanel, LogPanel, ProgressBar, Spinner, StatusBar};
pub use events::{Event, EventHandler};
pub use screens::{ComponentInfo, InstallScreen, MainScreen, OverviewData, Screen, StatusScreen};
