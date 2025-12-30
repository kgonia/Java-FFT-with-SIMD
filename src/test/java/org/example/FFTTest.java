package org.example;

import org.jtransforms.fft.DoubleFFT_1D;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class FFTTest {

    private static final double TOLERANCE = 1e-10;

    // Helper: convert IterativeFFT.Complex[] to separate real/imag arrays
    private static void complexToArrays(IterativeFFT.Complex[] c, double[] real, double[] imag) {
        for (int i = 0; i < c.length; i++) {
            real[i] = c[i].re;
            imag[i] = c[i].im;
        }
    }

    // Helper: create Complex array from real input (imaginary = 0)
    private static IterativeFFT.Complex[] realToComplex(double[] real) {
        IterativeFFT.Complex[] result = new IterativeFFT.Complex[real.length];
        for (int i = 0; i < real.length; i++) {
            result[i] = new IterativeFFT.Complex(real[i], 0);
        }
        return result;
    }

    // Helper: run JTransforms FFT (gold standard)
    private static void jtransformsFFT(double[] real, double[] imag) {
        int n = real.length;
        double[] interleaved = new double[2 * n];
        for (int i = 0; i < n; i++) {
            interleaved[2 * i] = real[i];
            interleaved[2 * i + 1] = imag[i];
        }

        DoubleFFT_1D fft = new DoubleFFT_1D(n);
        fft.complexForward(interleaved);

        for (int i = 0; i < n; i++) {
            real[i] = interleaved[2 * i];
            imag[i] = interleaved[2 * i + 1];
        }
    }

    // Helper: assert arrays are equal within tolerance
    private static void assertArraysEqual(double[] expected, double[] actual, String message) {
        assertEquals(expected.length, actual.length, message + " - length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOLERANCE,
                    message + " - mismatch at index " + i);
        }
    }

    // Helper: assert float arrays are equal within tolerance
    private static void assertFloatArraysEqual(float[] expected, float[] actual, float tolerance, String message) {
        assertEquals(expected.length, actual.length, message + " - length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance,
                    message + " - mismatch at index " + i);
        }
    }

    // ============ IterativeFFT Tests ============

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testIterativeFFT_Impulse(int n) {
        // Impulse input: [1, 0, 0, ..., 0]
        // Expected output: constant DC = 1 for all bins
        double[] realInput = new double[n];
        realInput[0] = 1.0;

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // IterativeFFT
        IterativeFFT.Complex[] iterData = realToComplex(realInput);
        IterativeFFT.fft(iterData);
        double[] iterReal = new double[n];
        double[] iterImag = new double[n];
        complexToArrays(iterData, iterReal, iterImag);

        assertArraysEqual(jtReal, iterReal, "IterativeFFT impulse real (n=" + n + ")");
        assertArraysEqual(jtImag, iterImag, "IterativeFFT impulse imag (n=" + n + ")");
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testIterativeFFT_SquareWave(int n) {
        // Square wave: first half = 1, second half = 0
        double[] realInput = new double[n];
        for (int i = 0; i < n / 2; i++) {
            realInput[i] = 1.0;
        }

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // IterativeFFT
        IterativeFFT.Complex[] iterData = realToComplex(realInput);
        IterativeFFT.fft(iterData);
        double[] iterReal = new double[n];
        double[] iterImag = new double[n];
        complexToArrays(iterData, iterReal, iterImag);

        assertArraysEqual(jtReal, iterReal, "IterativeFFT square wave real (n=" + n + ")");
        assertArraysEqual(jtImag, iterImag, "IterativeFFT square wave imag (n=" + n + ")");
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testIterativeFFT_SineWave(int n) {
        // Single frequency sine wave
        double[] realInput = new double[n];
        double freq = 4.0; // 4 cycles in the window
        for (int i = 0; i < n; i++) {
            realInput[i] = Math.sin(2 * Math.PI * freq * i / n);
        }

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // IterativeFFT
        IterativeFFT.Complex[] iterData = realToComplex(realInput);
        IterativeFFT.fft(iterData);
        double[] iterReal = new double[n];
        double[] iterImag = new double[n];
        complexToArrays(iterData, iterReal, iterImag);

        assertArraysEqual(jtReal, iterReal, "IterativeFFT sine wave real (n=" + n + ")");
        assertArraysEqual(jtImag, iterImag, "IterativeFFT sine wave imag (n=" + n + ")");
    }

    @Test
    void testIterativeFFT_Random() {
        int n = 512;
        Random rand = new Random(42); // Fixed seed for reproducibility
        double[] realInput = new double[n];
        double[] imagInput = new double[n];
        for (int i = 0; i < n; i++) {
            realInput[i] = rand.nextDouble() * 2 - 1; // [-1, 1]
            imagInput[i] = rand.nextDouble() * 2 - 1;
        }

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = imagInput.clone();
        jtransformsFFT(jtReal, jtImag);

        // IterativeFFT (with complex input)
        IterativeFFT.Complex[] iterData = new IterativeFFT.Complex[n];
        for (int i = 0; i < n; i++) {
            iterData[i] = new IterativeFFT.Complex(realInput[i], imagInput[i]);
        }
        IterativeFFT.fft(iterData);
        double[] iterReal = new double[n];
        double[] iterImag = new double[n];
        complexToArrays(iterData, iterReal, iterImag);

        assertArraysEqual(jtReal, iterReal, "IterativeFFT random complex real");
        assertArraysEqual(jtImag, iterImag, "IterativeFFT random complex imag");
    }

    // ============ VectorFFT Tests ============

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testVectorFFT_Impulse(int n) {
        // Impulse input: [1, 0, 0, ..., 0]
        double[] realInput = new double[n];
        realInput[0] = 1.0;

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT
        double[] vecReal = realInput.clone();
        double[] vecImag = new double[n];
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(jtReal, vecReal, "VectorFFT impulse real (n=" + n + ")");
        assertArraysEqual(jtImag, vecImag, "VectorFFT impulse imag (n=" + n + ")");
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testVectorFFT_SquareWave(int n) {
        // Square wave: first half = 1, second half = 0
        double[] realInput = new double[n];
        for (int i = 0; i < n / 2; i++) {
            realInput[i] = 1.0;
        }

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT
        double[] vecReal = realInput.clone();
        double[] vecImag = new double[n];
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(jtReal, vecReal, "VectorFFT square wave real (n=" + n + ")");
        assertArraysEqual(jtImag, vecImag, "VectorFFT square wave imag (n=" + n + ")");
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testVectorFFT_SineWave(int n) {
        // Single frequency sine wave
        double[] realInput = new double[n];
        double freq = 4.0; // 4 cycles in the window
        for (int i = 0; i < n; i++) {
            realInput[i] = Math.sin(2 * Math.PI * freq * i / n);
        }

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT
        double[] vecReal = realInput.clone();
        double[] vecImag = new double[n];
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(jtReal, vecReal, "VectorFFT sine wave real (n=" + n + ")");
        assertArraysEqual(jtImag, vecImag, "VectorFFT sine wave imag (n=" + n + ")");
    }

    @Test
    void testVectorFFT_Random() {
        int n = 512;
        Random rand = new Random(42); // Fixed seed for reproducibility
        double[] realInput = new double[n];
        double[] imagInput = new double[n];
        for (int i = 0; i < n; i++) {
            realInput[i] = rand.nextDouble() * 2 - 1; // [-1, 1]
            imagInput[i] = rand.nextDouble() * 2 - 1;
        }

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = imagInput.clone();
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT
        double[] vecReal = realInput.clone();
        double[] vecImag = imagInput.clone();
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(jtReal, vecReal, "VectorFFT random complex real");
        assertArraysEqual(jtImag, vecImag, "VectorFFT random complex imag");
    }

    // ============ Cross-comparison Tests ============

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testBothImplementations_Match(int n) {
        // Random input
        Random rand = new Random(123);
        double[] realInput = new double[n];
        for (int i = 0; i < n; i++) {
            realInput[i] = rand.nextDouble() * 2 - 1;
        }

        // IterativeFFT
        IterativeFFT.Complex[] iterData = realToComplex(realInput);
        IterativeFFT.fft(iterData);
        double[] iterReal = new double[n];
        double[] iterImag = new double[n];
        complexToArrays(iterData, iterReal, iterImag);

        // VectorFFT
        double[] vecReal = realInput.clone();
        double[] vecImag = new double[n];
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(iterReal, vecReal, "IterativeFFT vs VectorFFT real (n=" + n + ")");
        assertArraysEqual(iterImag, vecImag, "IterativeFFT vs VectorFFT imag (n=" + n + ")");
    }

    // ============ VectorFFT.fftReal Tests (vs JTransforms gold standard) ============

    private static final float FLOAT_TOLERANCE = 1e-5f;

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testFftRealFloat_Random(int n) {
        // Random real-only input
        Random rand = new Random(42);
        float[] realInput = new float[n];
        double[] realInputDouble = new double[n];
        for (int i = 0; i < n; i++) {
            realInput[i] = rand.nextFloat() * 2 - 1; // [-1, 1]
            realInputDouble[i] = realInput[i];
        }

        // JTransforms (gold standard)
        double[] jtReal = realInputDouble.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT.fftReal (float)
        float[] outReal = new float[n / 2 + 1];
        float[] outImag = new float[n / 2 + 1];
        VectorFFT.fftReal(realInput, outReal, outImag);

        // Compare first N/2+1 bins (unique values for real input)
        for (int i = 0; i <= n / 2; i++) {
            assertEquals((float) jtReal[i], outReal[i], FLOAT_TOLERANCE,
                    "fftReal float vs JTransforms real mismatch at bin " + i + " (n=" + n + ")");
            assertEquals((float) jtImag[i], outImag[i], FLOAT_TOLERANCE,
                    "fftReal float vs JTransforms imag mismatch at bin " + i + " (n=" + n + ")");
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testFftRealDouble_Random(int n) {
        // Random real-only input
        Random rand = new Random(42);
        double[] realInput = new double[n];
        for (int i = 0; i < n; i++) {
            realInput[i] = rand.nextDouble() * 2 - 1; // [-1, 1]
        }

        // JTransforms (gold standard)
        double[] jtReal = realInput.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT.fftReal (double)
        double[] outReal = new double[n / 2 + 1];
        double[] outImag = new double[n / 2 + 1];
        VectorFFT.fftReal(realInput, outReal, outImag);

        // Compare first N/2+1 bins
        for (int i = 0; i <= n / 2; i++) {
            assertEquals(jtReal[i], outReal[i], TOLERANCE,
                    "fftReal double vs JTransforms real mismatch at bin " + i + " (n=" + n + ")");
            assertEquals(jtImag[i], outImag[i], TOLERANCE,
                    "fftReal double vs JTransforms imag mismatch at bin " + i + " (n=" + n + ")");
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testFftRealFloat_Impulse(int n) {
        // Impulse: [1, 0, 0, ..., 0]
        float[] realInput = new float[n];
        realInput[0] = 1.0f;
        double[] realInputDouble = new double[n];
        realInputDouble[0] = 1.0;

        // JTransforms (gold standard)
        double[] jtReal = realInputDouble.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT.fftReal (float)
        float[] outReal = new float[n / 2 + 1];
        float[] outImag = new float[n / 2 + 1];
        VectorFFT.fftReal(realInput, outReal, outImag);

        for (int i = 0; i <= n / 2; i++) {
            assertEquals((float) jtReal[i], outReal[i], FLOAT_TOLERANCE,
                    "fftReal float impulse vs JTransforms real at bin " + i + " (n=" + n + ")");
            assertEquals((float) jtImag[i], outImag[i], FLOAT_TOLERANCE,
                    "fftReal float impulse vs JTransforms imag at bin " + i + " (n=" + n + ")");
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testFftRealFloat_SineWave(int n) {
        // Sine wave
        float[] realInput = new float[n];
        double[] realInputDouble = new double[n];
        double freq = 4.0;
        for (int i = 0; i < n; i++) {
            realInput[i] = (float) Math.sin(2 * Math.PI * freq * i / n);
            realInputDouble[i] = realInput[i];
        }

        // JTransforms (gold standard)
        double[] jtReal = realInputDouble.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT.fftReal (float)
        float[] outReal = new float[n / 2 + 1];
        float[] outImag = new float[n / 2 + 1];
        VectorFFT.fftReal(realInput, outReal, outImag);

        for (int i = 0; i <= n / 2; i++) {
            assertEquals((float) jtReal[i], outReal[i], FLOAT_TOLERANCE,
                    "fftReal float sine vs JTransforms real at bin " + i + " (n=" + n + ")");
            assertEquals((float) jtImag[i], outImag[i], FLOAT_TOLERANCE,
                    "fftReal float sine vs JTransforms imag at bin " + i + " (n=" + n + ")");
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {8, 16, 32, 64, 128, 256, 512, 1024})
    void testFftRealFloat_SquareWave(int n) {
        // Square wave: first half = 1, second half = 0
        float[] realInput = new float[n];
        double[] realInputDouble = new double[n];
        for (int i = 0; i < n / 2; i++) {
            realInput[i] = 1.0f;
            realInputDouble[i] = 1.0;
        }

        // JTransforms (gold standard)
        double[] jtReal = realInputDouble.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT.fftReal (float)
        float[] outReal = new float[n / 2 + 1];
        float[] outImag = new float[n / 2 + 1];
        VectorFFT.fftReal(realInput, outReal, outImag);

        for (int i = 0; i <= n / 2; i++) {
            assertEquals((float) jtReal[i], outReal[i], FLOAT_TOLERANCE,
                    "fftReal float square vs JTransforms real at bin " + i + " (n=" + n + ")");
            assertEquals((float) jtImag[i], outImag[i], FLOAT_TOLERANCE,
                    "fftReal float square vs JTransforms imag at bin " + i + " (n=" + n + ")");
        }
    }

    // ============ Edge Case Tests ============

    @Test
    void testMinimumSize() {
        // n = 2 is the minimum valid FFT size
        double[] real = {1.0, 0.0};
        double[] imag = {0.0, 0.0};

        // JTransforms
        double[] jtReal = real.clone();
        double[] jtImag = imag.clone();
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT
        double[] vecReal = real.clone();
        double[] vecImag = imag.clone();
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(jtReal, vecReal, "VectorFFT n=2 real");
        assertArraysEqual(jtImag, vecImag, "VectorFFT n=2 imag");

        // IterativeFFT
        IterativeFFT.Complex[] iterData = realToComplex(real);
        IterativeFFT.fft(iterData);
        double[] iterReal = new double[2];
        double[] iterImag = new double[2];
        complexToArrays(iterData, iterReal, iterImag);

        assertArraysEqual(jtReal, iterReal, "IterativeFFT n=2 real");
        assertArraysEqual(jtImag, iterImag, "IterativeFFT n=2 imag");
    }

    @Test
    void testDCSignal() {
        // All ones - should give DC component = N, rest = 0
        int n = 64;
        double[] real = new double[n];
        java.util.Arrays.fill(real, 1.0);

        // JTransforms
        double[] jtReal = real.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT
        double[] vecReal = real.clone();
        double[] vecImag = new double[n];
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(jtReal, vecReal, "VectorFFT DC signal real");
        assertArraysEqual(jtImag, vecImag, "VectorFFT DC signal imag");

        // Verify DC component is N
        assertEquals(n, vecReal[0], TOLERANCE, "DC component should be N");
    }

    @Test
    void testNyquistFrequency() {
        // Alternating +1, -1 pattern (Nyquist frequency)
        int n = 64;
        double[] real = new double[n];
        for (int i = 0; i < n; i++) {
            real[i] = (i % 2 == 0) ? 1.0 : -1.0;
        }

        // JTransforms
        double[] jtReal = real.clone();
        double[] jtImag = new double[n];
        jtransformsFFT(jtReal, jtImag);

        // VectorFFT
        double[] vecReal = real.clone();
        double[] vecImag = new double[n];
        VectorFFT.fft(vecReal, vecImag);

        assertArraysEqual(jtReal, vecReal, "VectorFFT Nyquist real");
        assertArraysEqual(jtImag, vecImag, "VectorFFT Nyquist imag");

        // IterativeFFT
        IterativeFFT.Complex[] iterData = realToComplex(real);
        IterativeFFT.fft(iterData);
        double[] iterReal = new double[n];
        double[] iterImag = new double[n];
        complexToArrays(iterData, iterReal, iterImag);

        assertArraysEqual(jtReal, iterReal, "IterativeFFT Nyquist real");
        assertArraysEqual(jtImag, iterImag, "IterativeFFT Nyquist imag");
    }
}
