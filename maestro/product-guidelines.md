# Rusty Stack - Product Guidelines

## Writing Style

### Prose Style
- **Primary Tone**: Technical, informative, playful and fun
- Balance technical accuracy with approachability
- Use clear, concise language while injecting personality where appropriate
- Make complex topics accessible without oversimplifying

### Voice & Personality
- **Professional & Approachable**: Maintain credibility while being friendly
- **Clear & Precise**: Avoid ambiguity in technical documentation
- **Witty & Engaging**: Use humor and wit to keep readers engaged

### Writing Guidelines
- Use active voice whenever possible
- Prefer simple sentence structures for complex concepts
- Use humor sparingly and appropriately (never at the expense of clarity)
- Embrace the "Rusty" theme in documentation where fun is appropriate
- Remember: "Code is like humor. When you have to explain it, it's bad!"

## Documentation Priorities

### 1. API Documentation (Highest Priority)
- Document all public interfaces comprehensively
- Include function signatures, parameters, return types, and usage examples
- Maintain API references for installer scripts, Python modules, and Rust code

### 2. Tutorials & How-To Guides
- Create step-by-step tutorials for common installation scenarios
- Cover troubleshooting and recovery procedures
- Include pre- and post-installation verification steps

### 3. Architecture Documentation
- Document the installer architecture (Rust TUI + Shell + Python)
- Explain the code mirroring pattern (`core/` ↔ `stans_ml_stack/core/`)
- Describe the multi-channel ROCm support system

## Error Messages

### Error Communication Principles
- **Detailed & Structured**: Provide comprehensive error information
- **User-Friendly**: Write messages that humans can understand
- **Actionable**: Include specific steps to resolve the issue
- **Referenced**: Link to relevant documentation or error codes

### Error Message Format
```
[ERROR] Component: <Component Name>
Message: <Clear, human-readable description>
Details: <Technical details for debugging>
Suggested Actions:
  - <Action 1>
  - <Action 2>
Documentation: <link or reference>
```

## Code Commenting Style

### When to Comment
- **Function Documentation**: Document purpose, parameters, return values
- **Complex Logic**: Explain non-obvious algorithms and reasoning
- **TODO Markers**: Flag future improvements and known limitations
- **Minimal Comments**: Let self-documenting code speak for itself when clear

### Comment Format (Python)
```python
def install_component(component: dict) -> bool:
    """
    Install a component using the appropriate installer script.

    Args:
        component: Dictionary with 'name', 'description', 'script', 'required' keys

    Returns:
        True if installation succeeded, False otherwise

    Raises:
        subprocess.CalledProcessError: If installer script fails
    """
    # TODO: Add support for interactive installation prompts
    ...
```

### Comment Format (Rust)
```rust
/// Run an installation script with the given arguments.
///
/// # Arguments
/// * `script` - Path to the installation script
/// * `args` - Command line arguments to pass to the script
///
/// # Returns
/// * `Ok(())` if script executed successfully
/// * `Err(Error)` if script execution failed
pub fn run_script(script: &Path, args: &[&str]) -> Result<()> {
    // Complex logic: Use sudo -S for non-interactive authentication
    // This is required because the TUI cannot control the terminal directly
    ...
}
```

## Visual Identity

### Thematic Elements
- **Rust/Spartan Theme**: Embrace strength, durability, and performance
- **AMD GPU Colors**: Red accents, dark backgrounds in terminal output
- **ASCII Art**: Use banners and decorative elements in the TUI installer

### Terminal Output Style
- Use color for highlighting (green for success, red for errors, yellow for warnings)
- Include progress indicators and spinners for long operations
- Show before/after benchmark comparisons when relevant

## Naming Conventions

### Components
- Installer: "Rusty-Stack" or "rusty-stack"
- Python Package: "stans-ml-stack" (backward compatibility)
- CLI Commands: `ml-stack-*` (e.g., `ml-stack-install`, `ml-stack-verify`)

### File Organization
- `scripts/` - Installation shell scripts
- `rusty-stack/` - Rust TUI installer
- `stans_ml_stack/` - Python package
- `maestro/` - Maestro project configuration
