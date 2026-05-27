fn main() {
    napi_build::setup();

    // Static OpenBLAS needs help linking into a cdylib (which tolerates
    // undefined symbols and so won't pull archive members on its own):
    //  - re-link libopenblas.a after the referencing objects so cblas_sgemm and
    //    friends are pulled from the archive;
    //  - link libpthread for OpenBLAS's threading runtime;
    //  - force `pthread_atfork`, which on aarch64 lives ONLY in libc_nonshared.a
    //    (x86_64 also exports it dynamically from libc.so.6, which is why x64
    //    linked fine); without -u it stays undefined and dlopen fails on arm64.
    // macOS uses Accelerate; this is Linux-only.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-l:libopenblas.a");
        println!("cargo:rustc-link-arg=-lpthread");
        println!("cargo:rustc-link-arg=-Wl,-u,pthread_atfork");
    }
}
