# JavaFFT with SIMD Support

This is just a Proof of Concept (PoC) of FFT in Java with SIMD support.

This is not intended to work as a standalone library.

The goal was not to measure performance against implementations in other languages.

JTransforms was used as a golden standard. Some tests are copied from JTransforms and adjusted.

Other tests check if results are the same/similar as in JTransforms.

Tests comparing the implementation for FFT on the real part show some discrepancies, but my goal was to make a good enough implementation, not an ideal one.
For complex results, they are the same (within tolerance).

---

Solution was inspired by this repository:

https://github.com/jagger2048/fft_simd

made by https://github.com/jagger2048

---

## Benchmark Results

*Note: These benchmark results were obtained on a specific machine configuration and may vary depending on the hardware and software environment.*

### System Architecture

```
Architecture:                x86_64
CPU op-mode(s):            32-bit, 64-bit
Address sizes:             40 bits physical, 48 bits virtual
Byte Order:                Little Endian
CPU(s):                      8
On-line CPU(s) list:       0-7
Vendor ID:                   GenuineIntel
Model name:                Intel Core Processor (Haswell, no TSX)
CPU family:              6
Model:                   60
Thread(s) per core:      1
Core(s) per socket:      1
```



The benchmark results below compare the performance of three different FFT implementations: `iterativeFFT` (a standard iterative approach), `vectorFFT` (an implementation leveraging SIMD, likely Project Panama Vector API), and `jtransforms` (used as a reference). The data indicates that `jtransforms` consistently outperforms both `iterativeFFT` and `vectorFFT` across various data sizes and precision types (double and float). The `vectorFFT` implementation, while aiming for performance with SIMD, appears slower than the `iterativeFFT` in these specific measurements.

### Results

```
FFTBenchmark.iterativeFFT_double     256  avgt   10     5.534 ±   0.026  us/op
FFTBenchmark.iterativeFFT_double    1024  avgt   10    30.407 ±   1.518  us/op
FFTBenchmark.iterativeFFT_double    4096  avgt   10   177.757 ±   9.703  us/op
FFTBenchmark.iterativeFFT_double   16384  avgt   10  1040.809 ±  15.065  us/op
FFTBenchmark.iterativeFFT_double   65536  avgt   10  4489.786 ±  63.985  us/op

FFTBenchmark.iterativeFFT_float      256  avgt   10     5.578 ±   0.109  us/op
FFTBenchmark.iterativeFFT_float     1024  avgt   10    30.565 ±   1.669  us/op
FFTBenchmark.iterativeFFT_float     4096  avgt   10   176.676 ±   2.237  us/op
FFTBenchmark.iterativeFFT_float    16384  avgt   10  1035.524 ±  20.260  us/op
FFTBenchmark.iterativeFFT_float    65536  avgt   10  4565.229 ± 113.500  us/op

FFTBenchmark.jtransforms_double      256  avgt   10     2.629 ±   0.023  us/op
FFTBenchmark.jtransforms_double     1024  avgt   10    12.407 ±   0.069  us/op
FFTBenchmark.jtransforms_double     4096  avgt   10    90.605 ±  11.098  us/op
FFTBenchmark.jtransforms_double    16384  avgt   10   295.572 ±  43.231  us/op
FFTBenchmark.jtransforms_double    65536  avgt   10   823.141 ±  34.211  us/op

FFTBenchmark.jtransforms_float       256  avgt   10     2.604 ±   0.018  us/op
FFTBenchmark.jtransforms_float      1024  avgt   10    12.608 ±   0.467  us/op
FFTBenchmark.jtransforms_float      4096  avgt   10    73.277 ±   9.360  us/op
FFTBenchmark.jtransforms_float     16384  avgt   10   247.372 ±  22.803  us/op
FFTBenchmark.jtransforms_float     65536  avgt   10   779.120 ±  41.660  us/op

FFTBenchmark.vectorFFT_double        256  avgt   10    12.828 ±   0.145  us/op
FFTBenchmark.vectorFFT_double       1024  avgt   10    60.350 ±   3.699  us/op
FFTBenchmark.vectorFFT_double       4096  avgt   10   278.969 ±  34.504  us/op
FFTBenchmark.vectorFFT_double      16384  avgt   10  1319.158 ± 158.907  us/op
FFTBenchmark.vectorFFT_double      65536  avgt   10  6912.764 ± 259.370  us/op

FFTBenchmark.vectorFFT_float         256  avgt   10    10.872 ±   0.312  us/op
FFTBenchmark.vectorFFT_float        1024  avgt   10    49.761 ±   3.119  us/op
FFTBenchmark.vectorFFT_float        4096  avgt   10   214.589 ±   8.080  us/op
FFTBenchmark.vectorFFT_float       16384  avgt   10  1021.880 ±  31.593  us/op
FFTBenchmark.vectorFFT_float       65536  avgt   10  4939.453 ± 554.655  us/op
```