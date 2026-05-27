fn main() {
    napi_build::setup();

    // openblas-src links libopenblas.a, but a cdylib tolerates undefined
    // symbols, so the linker leaves cblas_sgemm unresolved instead of pulling it
    // out of the archive (dlopen then fails: undefined symbol cblas_sgemm).
    // Re-link the static archive after the referencing objects so its cblas
    // members get pulled in. macOS uses Accelerate; this is Linux-only.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-l:libopenblas.a");
    }
}
