use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const BENCHMARK_RESULTS_MARKER: &str = "---BENCHMARK_RESULTS_START---";

/// Returns benchmark log directories mirroring the scripts/lib/benchmark_common.sh logic.
/// Precedence: MLSTACK_LOG_DIR (when usable), then $HOME/.rusty-stack/logs, then ${TMPDIR:-/tmp}/rusty-stack/logs.
pub fn benchmark_log_directories() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(env_dir) = env::var("MLSTACK_LOG_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            if let Some(dir) = existing_log_dir(&path) {
                push_unique(&mut dirs, dir);
            }
        }
    }

    for home in candidate_log_homes() {
        let path = Path::new(&home).join(".rusty-stack").join("logs");
        if let Some(dir) = existing_log_dir(&path) {
            push_unique(&mut dirs, dir);
        }
        // Codex/Claude environments often write benchmark logs here.
        let claude_tmp = Path::new(&home)
            .join(".claude")
            .join("tmp")
            .join("rusty-stack")
            .join("logs");
        if let Some(dir) = existing_log_dir(&claude_tmp) {
            push_unique(&mut dirs, dir);
        }
    }

    let tmp_root = env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    if !tmp_root.trim().is_empty() {
        let path = Path::new(&tmp_root).join("rusty-stack").join("logs");
        if let Some(dir) = existing_log_dir(&path) {
            push_unique(&mut dirs, dir);
        }
    }

    dirs
}

/// Collects log files that contain any of the provided patterns, sorted by modification time.
pub fn collect_matching_logs(log_dirs: &[PathBuf], patterns: &[&str]) -> Vec<PathBuf> {
    let mut matches = Vec::new();
    for dir in log_dirs {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.filter_map(|entry| entry.ok()) {
                let name = entry.file_name().to_string_lossy().into_owned();
                if patterns.iter().any(|pat| name.contains(pat)) {
                    let path = entry.path();
                    if path.is_file() {
                        matches.push(path);
                    }
                }
            }
        }
    }

    matches.sort_by(|a, b| file_modified(a).cmp(&file_modified(b)));
    matches
}

/// Finds the most recent log file across the provided directories that matches the pattern.
/// If any `.json` file is available, prefer the most recent JSON result.
pub fn find_latest_log_in_dirs(log_dirs: &[PathBuf], pattern: &str) -> Option<PathBuf> {
    let matches = collect_matching_logs(log_dirs, &[pattern]);
    matches
        .iter()
        .filter(|path| is_json_log(path))
        .max_by_key(|path| file_modified(path))
        .cloned()
        .or_else(|| matches.last().cloned())
}

pub fn is_json_log(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

/// Extract a benchmark JSON object from raw log content.
/// Handles plain JSON files, marker-delimited output, and mixed logs.
pub fn extract_benchmark_json_value(contents: &str) -> Option<serde_json::Value> {
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Some(value);
    }

    // Prefer marker matches that appear at line-start to avoid marker-like text inside JSON strings.
    let marker_positions: Vec<usize> = contents
        .match_indices(BENCHMARK_RESULTS_MARKER)
        .map(|(idx, _)| idx)
        .filter(|idx| marker_is_line_start(contents, *idx))
        .collect();

    for marker_idx in marker_positions.into_iter().rev() {
        let search = &contents[marker_idx + BENCHMARK_RESULTS_MARKER.len()..];
        if let Some(value) = parse_best_json_object(search) {
            return Some(value);
        }
    }

    parse_best_json_object(contents)
}

/// Extract benchmark JSON as a normalized string.
pub fn extract_benchmark_json_string(contents: &str) -> Option<String> {
    extract_benchmark_json_value(contents)
        .and_then(|value| serde_json::to_string(&value).ok())
}

fn marker_is_line_start(contents: &str, marker_idx: usize) -> bool {
    if marker_idx == 0 {
        return true;
    }
    contents[..marker_idx].ends_with('\n')
}

fn parse_best_json_object(input: &str) -> Option<serde_json::Value> {
    let mut best: Option<(usize, usize, serde_json::Value)> = None;

    for (idx, ch) in input.char_indices() {
        if ch != '{' {
            continue;
        }
        let Some(end) = find_balanced_json_object_end(input, idx) else {
            continue;
        };

        let candidate = &input[idx..end];
        let Ok(value) = serde_json::from_str::<serde_json::Value>(candidate) else {
            continue;
        };

        let score = benchmark_json_score(&value);
        let candidate_len = end.saturating_sub(idx);
        let is_better = best
            .as_ref()
            .map(|(best_score, best_len, _)| {
                score > *best_score || (score == *best_score && candidate_len > *best_len)
            })
            .unwrap_or(true);
        if is_better {
            best = Some((score, candidate_len, value));
        }
    }

    best.map(|(_, _, value)| value)
}

fn benchmark_json_score(value: &serde_json::Value) -> usize {
    let mut score = 0usize;
    if let Some(obj) = value.as_object() {
        if obj.get("results").and_then(|v| v.as_object()).is_some() {
            score += 200;
        }
        if obj.get("metrics").and_then(|v| v.as_object()).is_some() {
            score += 100;
        }
        if obj.get("errors").and_then(|v| v.as_array()).is_some() {
            score += 20;
        }
        if obj.get("success").and_then(|v| v.as_bool()).is_some() {
            score += 15;
        }
        if obj.get("name").and_then(|v| v.as_str()).is_some() {
            score += 10;
        }
        score += obj.len();
    }

    score
}

fn find_balanced_json_object_end(input: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in input[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return Some(start + offset + 1);
                }
            }
            _ => {}
        }
    }

    None
}

fn existing_log_dir(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        Some(path.to_path_buf())
    } else {
        None
    }
}

fn user_home_from_passwd(user_name: &str) -> Option<String> {
    let target = user_name.trim();
    if target.is_empty() {
        return None;
    }
    let passwd = fs::read_to_string("/etc/passwd").ok()?;
    passwd.lines().find_map(|line| {
        if line.trim().is_empty() || line.starts_with('#') {
            return None;
        }
        let fields: Vec<&str> = line.split(':').collect();
        if fields.len() < 6 || fields[0] != target {
            return None;
        }
        let home = fields[5].trim();
        if home.is_empty() {
            None
        } else {
            Some(home.to_string())
        }
    })
}

fn candidate_log_homes() -> Vec<String> {
    let mut homes = Vec::new();

    if let Ok(home) = env::var("MLSTACK_USER_HOME") {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            homes.push(trimmed.to_string());
        }
    }
    if let Ok(home) = env::var("HOME") {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            homes.push(trimmed.to_string());
        }
    }
    for key in ["SUDO_USER", "USER", "LOGNAME"] {
        if let Ok(user_name) = env::var(key) {
            if let Some(home) = user_home_from_passwd(&user_name) {
                homes.push(home);
            }
        }
    }

    if let Ok(passwd) = fs::read_to_string("/etc/passwd") {
        for line in passwd.lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split(':').collect();
            if fields.len() < 6 {
                continue;
            }
            let home = fields[5].trim();
            if home.is_empty() {
                continue;
            }
            let home_path = Path::new(home);
            if home_path.join(".mlstack_env").is_file() || home_path.join(".mlstack").is_dir() {
                homes.push(home.to_string());
            }
        }
    }

    homes.sort();
    homes.dedup();
    homes
}

fn push_unique(dirs: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !dirs.iter().any(|existing| existing == &candidate) {
        dirs.push(candidate);
    }
}

fn file_modified(path: &PathBuf) -> Option<std::time::SystemTime> {
    fs::metadata(path)
        .ok()
        .and_then(|meta| meta.modified().ok())
}
