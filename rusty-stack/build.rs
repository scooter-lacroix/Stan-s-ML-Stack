fn main() {
    if let Ok(token) = std::env::var("GITHUB_INSTALLER_TOKEN") {
        println!("cargo:rustc-env=GITHUB_INSTALLER_TOKEN={token}");
    }
}
