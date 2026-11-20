# SIMD build selection

`configure` picks the SIMD mode at configure time. There is no runtime dispatch.

Requirement: `configure.ac` reads the upstream `../utility/structures` directory, which sits beside `changa/` in the ChaNGa source tree. This repo does not include it, matching the pristine fork. It is only needed to build and test, not to run on a cluster. Test builds need a `utility/` tree next to `changa/`, for example copied from an upstream checkout.

Hard requirement: `config.guess` and `config.sub` must stay in the repo root. The generated `configure` checks for them and exits with an error if they are missing. They are tracked through `!config.guess` and `!config.sub` in `.gitignore`.

| `--enable-arch` | Target flags | Lanes (double / float) |
|---|---|---|
| `auto` (default) | `native` if the compiler has C++17 `std::experimental::simd`; otherwise `legacy-avx` (AVX host) or `legacy-sse2` (SSE2 host) | host-dependent |
| `native` | `-march=native -mprefer-vector-width=512` | widest the host supports |
| `avx512` | `-march=native -mavx512f -mavx512dq -mprefer-vector-width=512` | 8 / 16 |
| `avx` | `-march=native -mno-avx512f` | 4 / 8 |
| `sse2` | `-march=native -mno-avx` | 2 / 4 |
| `neon` | AArch64 `-march=armv8-a`; ARMv7 `-march=armv7-a -mfpu=neon` | AArch64 2 / 4; ARMv7 no double SIMD, 4 float |
| `legacy-sse2` | `-march=native -mno-avx -DCMK_USE_SSE2` (fork Charm intrinsics) | 2 / 4 |
| `legacy-avx` | `-march=native -mno-avx512f -DCMK_USE_AVX` (fork Charm intrinsics) | 4 / 8 |

- `auto` first probes for C++17 `std::experimental::simd` (with `-std=c++17
  -march=native`). If it compiles, `auto` selects `native`, which gives the
  widest vectors the host supports (AVX-512, AVX, SSE2, or NEON). If it does
  not, `auto` falls back to `legacy-avx` when the host has AVX, else to
  `legacy-sse2` on x86 with SSE2. A host with neither stops configure with an
  error. Float is never chosen automatically.
- The fallback also sets the C++ standard. With C++17 available, `cxx_std` is
  `c++17`. Without it, `cxx_std` is `c++11`, the fork's standard, and only the
  `legacy-*` modes can build. Explicit `native`, `avx`, `avx512`, `sse2`, or
  `neon` on a compiler without C++17 stops configure with an error.
- Explicit modes never silently downgrade. An explicit mode the host or
  compiler cannot support, such as `avx512` on a host without AVX-512F, stops
  configure with an error that names the missing requirement. `none` and other
  unknown names are also errors.
- `--enable-float` adds `-DCOSMO_FLOAT`. Opt-in only.
- The deprecated `--enable-sse2` and `--enable-avx` flags are read only
  when `--enable-arch` is not given. They map to `legacy-sse2` and
  `legacy-avx`, the fork's intrinsics path, so existing scripts keep their
  old behaviour.
- `-ffp-contract=off` is always set. It keeps `a*b+c` unfused, so output
  stays bit-exact with the fork, which uses no FMA.
- The probe prints `auto: selected SIMD mode is ...` during configure.

Examples:

```sh
./configure                          # auto
./configure --enable-arch=native     # widest the host supports
./configure --enable-arch=sse2       # override: SSE2 double
./configure --enable-arch=avx --enable-float
```

Kernel selection: the fork's Charm intrinsics kernels in `gravity.h` and
`SSEdefs.h` are compiled only when `CMK_SSE` is defined, which the
`legacy-*` modes set. All other modes compile the native
`std::experimental::simd` kernels (guarded by `#if !CMK_SSE`).

Charm++ prerequisites: a Charm++ build with `charmc`. The native modes need a
C++17 compiler that provides `<experimental/simd>` (GCC 11 or newer). The
`legacy-*` modes need only C++11. `charmc` hardcodes `-std=gnu++11` in
`OPTS_CXX`, so `Makefile.in` passes the standard again through
`-c++-option -std=$(cxx_std)`, which is applied after `OPTS_CXX`.
`MapStructures.h` no longer has `using namespace std;`, because
`std::byte` clashes with Charm's global `byte` typedef in `liveViz0.h`.
