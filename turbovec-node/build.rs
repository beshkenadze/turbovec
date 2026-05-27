fn main() {
    napi_build::setup();

    // openblas-src links libopenblas.a, but a cdylib tolerates undefined
    // symbols, so the linker leaves the archive's references unresolved rather
    // than pulling members in (dlopen then fails: first `cblas_sgemm`, then
    // `pthread_atfork`). Re-link the static archive after the referencing
    // objects to pull its members, then link libpthread — OpenBLAS's threading
    // runtime, which openblas-src does not add to its extra libs. Order matters:
    // pthread comes after the archive that references it. macOS uses Accelerate;
    // this is Linux-only.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-l:libopenblas.a");
        println!("cargo:rustc-link-arg=-lpthread");
    }
}
