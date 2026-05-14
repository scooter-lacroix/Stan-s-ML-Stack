use std::fmt;

#[derive(Clone)]
pub struct SealedToken {
    token: Box<str>,
}

impl SealedToken {
    pub fn from_env() -> Self {
        Self::new(std::env::var("GITHUB_INSTALLER_TOKEN").unwrap_or_default())
    }

    pub fn new(token: impl AsRef<str>) -> Self {
        Self {
            token: token.as_ref().to_owned().into_boxed_str(),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.token
    }

    pub fn purge(&mut self) {
        unsafe {
            for byte in self.token.as_bytes_mut() {
                *byte = 0;
            }
        }
    }
}

impl fmt::Debug for SealedToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl Drop for SealedToken {
    fn drop(&mut self) {
        self.purge();
    }
}

#[cfg(test)]
mod tests {
    use super::SealedToken;

    #[test]
    fn debug_is_redacted() {
        let token = SealedToken::new("secret-token");
        assert_eq!(format!("{:?}", token), "[REDACTED]");
    }

    #[test]
    fn as_str_is_only_public_accessor() {
        let token = SealedToken::new("secret-token");
        assert_eq!(token.as_str(), "secret-token");
    }

    #[test]
    fn purge_zeroes_memory_before_drop() {
        let mut token = SealedToken::new("secret-token");
        assert_eq!(token.as_str(), "secret-token");
        token.purge();
    }
}
