fn main() {
    // Compile-time enforcement: env!() fails the build if the env var is absent.
    // The value is also re-exported via cargo:rustc-env so it is accessible at runtime
    // via std::env::var("GITHUB_INSTALLER_TOKEN") without any source-file string literal.
    let token = env!("GITHUB_INSTALLER_TOKEN");
    println!("cargo:rustc-env=GITHUB_INSTALLER_TOKEN={}", token);
}
