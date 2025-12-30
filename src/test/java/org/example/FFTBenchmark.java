package org.example;

import org.jtransforms.fft.DoubleFFT_1D;
import org.jtransforms.fft.FloatFFT_1D;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmark comparing FFT implementations:
 * - JTransforms (reference library)
 * - VectorFFT (SIMD Vector API implementation)
 * - IterativeFFT (simple iterative implementation)
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 2, jvmArgs = {"--add-modules", "jdk.incubator.vector"})
public class FFTBenchmark {

    @Param({"256", "1024", "4096", "16384", "65536"})
    private int size;

    // JTransforms data (interleaved)
    private double[] jtransformsDataDouble;
    private float[] jtransformsDataFloat;
    private DoubleFFT_1D jtransformsDoubleFFT;
    private FloatFFT_1D jtransformsFloatFFT;

    // VectorFFT data (split real/imag)
    private double[] vectorRealDouble;
    private double[] vectorImagDouble;
    private float[] vectorRealFloat;
    private float[] vectorImagFloat;

    // IterativeFFT data (Complex objects)
    private IterativeFFT.Complex[] iterativeDataDouble;
    private IterativeFFT.ComplexF[] iterativeDataFloat;

    // Source data for reset
    private double[] sourceRealDouble;
    private double[] sourceImagDouble;
    private float[] sourceRealFloat;
    private float[] sourceImagFloat;

    @Setup(Level.Trial)
    public void setup() {
        Random random = new Random(42);

        // Generate random complex data
        sourceRealDouble = new double[size];
        sourceImagDouble = new double[size];
        sourceRealFloat = new float[size];
        sourceImagFloat = new float[size];

        for (int i = 0; i < size; i++) {
            sourceRealDouble[i] = 2.0 * random.nextDouble() - 1.0;
            sourceImagDouble[i] = 2.0 * random.nextDouble() - 1.0;
            sourceRealFloat[i] = (float) sourceRealDouble[i];
            sourceImagFloat[i] = (float) sourceImagDouble[i];
        }

        // Initialize JTransforms
        jtransformsDoubleFFT = new DoubleFFT_1D(size);
        jtransformsFloatFFT = new FloatFFT_1D(size);
        jtransformsDataDouble = new double[2 * size];
        jtransformsDataFloat = new float[2 * size];

        // Initialize VectorFFT arrays
        vectorRealDouble = new double[size];
        vectorImagDouble = new double[size];
        vectorRealFloat = new float[size];
        vectorImagFloat = new float[size];

        // Initialize IterativeFFT arrays
        iterativeDataDouble = new IterativeFFT.Complex[size];
        iterativeDataFloat = new IterativeFFT.ComplexF[size];
        for (int i = 0; i < size; i++) {
            iterativeDataDouble[i] = new IterativeFFT.Complex(0, 0);
            iterativeDataFloat[i] = new IterativeFFT.ComplexF(0, 0);
        }
    }

    @Setup(Level.Invocation)
    public void resetData() {
        // Reset JTransforms data (interleaved format)
        for (int i = 0; i < size; i++) {
            jtransformsDataDouble[2 * i] = sourceRealDouble[i];
            jtransformsDataDouble[2 * i + 1] = sourceImagDouble[i];
            jtransformsDataFloat[2 * i] = sourceRealFloat[i];
            jtransformsDataFloat[2 * i + 1] = sourceImagFloat[i];
        }

        // Reset VectorFFT data
        System.arraycopy(sourceRealDouble, 0, vectorRealDouble, 0, size);
        System.arraycopy(sourceImagDouble, 0, vectorImagDouble, 0, size);
        System.arraycopy(sourceRealFloat, 0, vectorRealFloat, 0, size);
        System.arraycopy(sourceImagFloat, 0, vectorImagFloat, 0, size);

        // Reset IterativeFFT data
        for (int i = 0; i < size; i++) {
            iterativeDataDouble[i].re = sourceRealDouble[i];
            iterativeDataDouble[i].im = sourceImagDouble[i];
            iterativeDataFloat[i].re = sourceRealFloat[i];
            iterativeDataFloat[i].im = sourceImagFloat[i];
        }
    }

    // ==================== Double Precision Benchmarks ====================

    @Benchmark
    public void jtransforms_double(org.openjdk.jmh.infra.Blackhole bh) {
        jtransformsDoubleFFT.complexForward(jtransformsDataDouble);
        bh.consume(jtransformsDataDouble);
    }

    @Benchmark
    public void vectorFFT_double(org.openjdk.jmh.infra.Blackhole bh) {
        VectorFFT.fft(vectorRealDouble, vectorImagDouble);
        bh.consume(vectorRealDouble);
        bh.consume(vectorImagDouble);
    }

    @Benchmark
    public void iterativeFFT_double(org.openjdk.jmh.infra.Blackhole bh) {
        IterativeFFT.fft(iterativeDataDouble);
        bh.consume(iterativeDataDouble);
    }

    // ==================== Float Precision Benchmarks ====================

    @Benchmark
    public void jtransforms_float(org.openjdk.jmh.infra.Blackhole bh) {
        jtransformsFloatFFT.complexForward(jtransformsDataFloat);
        bh.consume(jtransformsDataFloat);
    }

    @Benchmark
    public void vectorFFT_float(org.openjdk.jmh.infra.Blackhole bh) {
        VectorFFT.fft(vectorRealFloat, vectorImagFloat);
        bh.consume(vectorRealFloat);
        bh.consume(vectorImagFloat);
    }

    @Benchmark
    public void iterativeFFT_float(org.openjdk.jmh.infra.Blackhole bh) {
        IterativeFFT.fft(iterativeDataFloat);
        bh.consume(iterativeDataFloat);
    }

    // ==================== Main Method ====================

    static void main() throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(FFTBenchmark.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
