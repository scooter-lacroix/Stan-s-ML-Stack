use std::fmt;

#[derive(Clone)]
pub struct SealedToken(String);

impl SealedToken {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn from_env() -> Self {
        Self(std::env::var("GITHUB_INSTALLER_TOKEN").unwrap_or_default())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn purge(&mut self) {
        let len = self.0.len();
        self.0.clear();
        self.0.extend(std::iter::repeat_n('\0', len));
        self.0.clear();
    }
}

impl fmt::Debug for SealedToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_is_redacted() {
        let token = SealedToken::new("abc");
        assert_eq!(format!("{:?}", token), "[REDACTED]");
    }
}
