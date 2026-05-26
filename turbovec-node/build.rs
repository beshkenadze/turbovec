fn main() {
    napi_build::setup();

    // The core `turbovec` crate calls into OpenBLAS (e.g. cblas_sgemm) on Linux
    // through ndarray's `blas` feature. The linker's default --as-needed drops
    // libopenblas from the addon's NEEDED list because it appears ahead of the
    // objects that reference it in link order, yielding a `.node` that fails at
    // dlopen with `undefined symbol: cblas_sgemm`. Re-link it after those
    // objects so the reference resolves. macOS uses Accelerate and needs nothing.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-lopenblas");
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
    }
}
