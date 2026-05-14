fn main() {
    // Compile-time enforcement: fail immediately if GITHUB_INSTALLER_TOKEN is absent.
    // The env!() macro reads the variable at compile time and embeds its value in the
    // binary's compiled environment, making it available via std::env::var() at runtime.
    // This is the standard rustc mechanism for build-script-to-binary token injection.
    let _token = env!("GITHUB_INSTALLER_TOKEN");
}
