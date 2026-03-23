use chrono::Local;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// A single message in the conversation (user or assistant).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

/// Shared application state: conversation history + SSE broadcast channel.
#[derive(Clone)]
pub struct AppState {
    pub history: Arc<Mutex<Vec<ChatMsg>>>,
    pub tx: broadcast::Sender<ChatMsg>,
}

impl AppState {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(64);
        Self {
            history: Arc::new(Mutex::new(Vec::new())),
            tx,
        }
    }

    /// Append a message to history and broadcast it to all SSE subscribers.
    pub fn push(&self, role: &str, content: &str) {
        let msg = ChatMsg {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Local::now().format("%H:%M:%S").to_string(),
        };
        self.history.lock().unwrap().push(msg.clone());
        let _ = self.tx.send(msg);
    }
}
