use std::fs;
use std::path::Path;
use std::env;

fn main() -> std::io::Result<()> {
    let home = env::var("HOME").expect("HOME env var not set");
    let current_dir = env::current_dir()?;
    
    // Pattern to replace
    let old_path = "$HOME";
    // Replacement (dynamic $HOME)
    let new_path = "$HOME";
    
    // We'll walk through scripts/ and rusty-stack/src/
    let dirs = vec!["scripts", "rusty-stack/src"];
    
    for dir in dirs {
        let path = current_dir.join(dir);
        if !path.exists() { continue; }
        
        process_dir(&path, old_path, new_path)?;
    }
    
    // Also process .mlstack_env if it exists in HOME
    let env_file = Path::new(&home).join(".mlstack_env");
    if env_file.exists() {
        process_file(&env_file, old_path, new_path)?;
    }

    println!("Portability fix applied: Replaced {} with {} in all relevant files.", old_path, new_path);
    Ok(())
}

fn process_dir(dir: &Path, old: &str, new: &str) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            process_dir(&path, old, new)?;
        } else if path.is_file() {
            let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if extension == "sh" || extension == "py" || extension == "rs" || extension == "json" || path.file_name().unwrap() == ".mlstack_env" {
                process_file(&path, old, new)?;
            }
        }
    }
    Ok(())
}

fn process_file(path: &Path, old: &str, new: &str) -> std::io::Result<()> {
    let content = fs::read_to_string(path)?;
    if content.contains(old) {
        println!("Patching file: {:?}", path);
        // Special case for Rust files: replace with env::var("HOME") logic or similar if needed
        // but for now, the user asked for agnostic pathing.
        // Actually, for .rs files, we should probably use a dynamic approach.
        let new_content = if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            // Replace hardcoded $HOME/ with a logic that uses the home variable if we're in a string
            // But let's look at the grep results for .rs files first.
            // Grep showed no .rs files with $HOME! Good.
            content.replace(old, new)
        } else {
            content.replace(old, new)
        };
        fs::write(path, new_content)?;
    }
    Ok(())
}
