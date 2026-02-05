//! Event Handling
//!
//! Keyboard and resize event handling.

use crossterm::event::{self, Event as CrosstermEvent, KeyEvent};
use std::time::Duration;
use tokio::sync::mpsc;

/// Application events.
#[derive(Debug, Clone)]
pub enum Event {
    /// Keyboard event
    Key(KeyEvent),
    /// Terminal resize
    Resize(u16, u16),
    /// Tick for animations
    Tick,
}

/// Event handler for async event processing.
pub struct EventHandler {
    /// Event receiver
    rx: mpsc::UnboundedReceiver<Event>,
    /// Tick rate
    _tick_rate: Duration,
}

impl EventHandler {
    /// Creates a new event handler.
    pub fn new(tick_rate: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn event polling task
        tokio::spawn(async move {
            loop {
                // Check for crossterm events
                if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                    match event::read() {
                        Ok(CrosstermEvent::Key(key)) => {
                            let _ = tx.send(Event::Key(key));
                        }
                        Ok(CrosstermEvent::Resize(w, h)) => {
                            let _ = tx.send(Event::Resize(w, h));
                        }
                        _ => {}
                    }
                }

                // Send tick event
                let _ = tx.send(Event::Tick);
                tokio::time::sleep(tick_rate).await;
            }
        });

        Self {
            rx,
            _tick_rate: tick_rate,
        }
    }

    /// Gets the next event with timeout.
    pub async fn next_event(&mut self, timeout: Duration) -> Option<Event> {
        tokio::time::timeout(timeout, self.rx.recv())
            .await
            .ok()
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_handler_creation() {
        let handler = EventHandler::new(Duration::from_millis(250));
        // Just verify it creates without panic
        let _ = handler;
    }
}
