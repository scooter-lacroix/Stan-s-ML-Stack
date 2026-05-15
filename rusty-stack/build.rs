fn main() {
    let token = std::env::var("GITHUB_INSTALLER_TOKEN")
        .expect("GITHUB_INSTALLER_TOKEN missing");
    println!("cargo:rustc-env=GITHUB_INSTALLER_TOKEN={token}");
}
