# JavaFFT with SIMD Support

This is just a Proof of Concept (PoC) of FFT in Java with SIMD support.

This is not intended to work as a standalone library.

The goal was not to measure performance against implementations in other languages.

JTransforms was used as a golden standard. Some tests are copied from JTransforms and adjusted.

Other tests check if results are the same/similar as in JTransforms.

Tests comparing the implementation for FFT on the real part show some discrepancies, but my goal was to make a good enough implementation, not an ideal one.
For complex results, they are the same (within tolerance).

After all it seems that SIMD usage isn't a crucial factor for performance. After first tests I did improvements but they aren't around SIMD usage. 
I don't want to work more on this project, so I leave it as is.

iterative FFT wasn't optimized after with naive implementation, so there is still room for improvements.
---

Solution was inspired by this repository:

https://github.com/jagger2048/fft_simd

made by https://github.com/jagger2048

---

## Benchmark Results

*Note: These benchmark results were obtained on a specific machine configuration and may vary depending on the hardware and software environment.*

```agsl
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   12
  On-line CPU(s) list:    0-11
Vendor ID:                GenuineIntel
  Model name:             11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
    CPU family:           6
    Model:                141
    Thread(s) per core:   2
    Core(s) per socket:   6
```

```agsl
Benchmark                         (size)  Mode  Cnt     Score     Error  Units
FFTBenchmark.iterativeFFT_double     256  avgt   10     3.240 ±   0.059  us/op
FFTBenchmark.iterativeFFT_double    1024  avgt   10    16.036 ±   0.197  us/op
FFTBenchmark.iterativeFFT_double    4096  avgt   10    83.640 ±   3.747  us/op
FFTBenchmark.iterativeFFT_double   16384  avgt   10   424.555 ±   6.109  us/op
FFTBenchmark.iterativeFFT_double   65536  avgt   10  2144.883 ± 229.283  us/op

FFTBenchmark.iterativeFFT_float      256  avgt   10     3.140 ±   0.026  us/op
FFTBenchmark.iterativeFFT_float     1024  avgt   10    15.860 ±   0.433  us/op
FFTBenchmark.iterativeFFT_float     4096  avgt   10    80.348 ±   2.560  us/op
FFTBenchmark.iterativeFFT_float    16384  avgt   10   429.668 ±  10.591  us/op
FFTBenchmark.iterativeFFT_float    65536  avgt   10  2097.029 ± 113.702  us/op

FFTBenchmark.jtransforms_double      256  avgt   10     1.367 ±   0.046  us/op
FFTBenchmark.jtransforms_double     1024  avgt   10     6.023 ±   0.093  us/op
FFTBenchmark.jtransforms_double     4096  avgt   10    35.426 ±   1.288  us/op
FFTBenchmark.jtransforms_double    16384  avgt   10   122.288 ±  23.547  us/op
FFTBenchmark.jtransforms_double    65536  avgt   10   384.217 ±  39.547  us/op

FFTBenchmark.jtransforms_float       256  avgt   10     1.415 ±   0.046  us/op
FFTBenchmark.jtransforms_float      1024  avgt   10     6.421 ±   0.434  us/op
FFTBenchmark.jtransforms_float      4096  avgt   10    34.259 ±   1.177  us/op
FFTBenchmark.jtransforms_float     16384  avgt   10   113.458 ±  11.994  us/op
FFTBenchmark.jtransforms_float     65536  avgt   10   335.568 ±   2.844  us/op

FFTBenchmark.vectorFFT_double        256  avgt   10     1.713 ±   0.132  us/op
FFTBenchmark.vectorFFT_double       1024  avgt   10     7.387 ±   0.170  us/op
FFTBenchmark.vectorFFT_double       4096  avgt   10    34.155 ±   1.405  us/op
FFTBenchmark.vectorFFT_double      16384  avgt   10   154.069 ±   5.312  us/op
FFTBenchmark.vectorFFT_double      65536  avgt   10   751.163 ±  53.198  us/op

FFTBenchmark.vectorFFT_float         256  avgt   10     1.648 ±   0.209  us/op
FFTBenchmark.vectorFFT_float        1024  avgt   10     6.073 ±   0.160  us/op
FFTBenchmark.vectorFFT_float        4096  avgt   10    27.106 ±   1.029  us/op
FFTBenchmark.vectorFFT_float       16384  avgt   10   116.167 ±   2.159  us/op
FFTBenchmark.vectorFFT_float       65536  avgt   10   495.448 ±  32.406  us/op
```