package org.example;

import jdk.incubator.vector.*;
import java.util.concurrent.ConcurrentHashMap;

public class VectorFFT {

    static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    static final VectorSpecies<Float> SPECIES_FLOAT = FloatVector.SPECIES_PREFERRED;

    // ==================== CACHED TWIDDLE TABLES ====================

    // Radix-4 twiddle cache: [cos1, sin1, cos2, sin2, cos3, sin3] per stage
    private static final ConcurrentHashMap<Integer, double[][]> TWIDDLE_CACHE_R4_D = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Integer, float[][]> TWIDDLE_CACHE_R4_F = new ConcurrentHashMap<>();
    // Radix-2 for odd log2(n) final stage
    private static final ConcurrentHashMap<Integer, double[][]> TWIDDLE_CACHE_R2_D = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Integer, float[][]> TWIDDLE_CACHE_R2_F = new ConcurrentHashMap<>();
    // fftReal unpack twiddles
    private static final ConcurrentHashMap<Integer, double[][]> UNPACK_TWIDDLE_D = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Integer, float[][]> UNPACK_TWIDDLE_F = new ConcurrentHashMap<>();

    // Temp buffer pools (per-thread)
    private static final ThreadLocal<double[]> TEMP_REAL_D = new ThreadLocal<>();
    private static final ThreadLocal<double[]> TEMP_IMAG_D = new ThreadLocal<>();
    private static final ThreadLocal<float[]> TEMP_REAL_F = new ThreadLocal<>();
    private static final ThreadLocal<float[]> TEMP_IMAG_F = new ThreadLocal<>();

    // ==================== TWIDDLE COMPUTATION ====================

    private static double[][] getRadix4TwiddlesDouble(int n) {
        return TWIDDLE_CACHE_R4_D.computeIfAbsent(n, VectorFFT::computeRadix4TwiddlesDouble);
    }

    private static float[][] getRadix4TwiddlesFloat(int n) {
        return TWIDDLE_CACHE_R4_F.computeIfAbsent(n, VectorFFT::computeRadix4TwiddlesFloat);
    }

    private static double[][] getRadix2TwiddlesDouble(int n) {
        return TWIDDLE_CACHE_R2_D.computeIfAbsent(n, VectorFFT::computeRadix2TwiddlesDouble);
    }

    private static float[][] getRadix2TwiddlesFloat(int n) {
        return TWIDDLE_CACHE_R2_F.computeIfAbsent(n, VectorFFT::computeRadix2TwiddlesFloat);
    }

    private static double[][] computeRadix4TwiddlesDouble(int n) {
        int numStages = Integer.numberOfTrailingZeros(n) / 2;
        // 6 arrays per stage: cos1, sin1, cos2, sin2, cos3, sin3
        double[][] twiddles = new double[numStages * 6][];

        for (int s = 0; s < numStages; s++) {
            int m = 1 << ((s + 1) * 2); // 4, 16, 64, ...
            int quarterM = m / 4;
            double[] cos1 = new double[quarterM];
            double[] sin1 = new double[quarterM];
            double[] cos2 = new double[quarterM];
            double[] sin2 = new double[quarterM];
            double[] cos3 = new double[quarterM];
            double[] sin3 = new double[quarterM];

            double baseAngle = -2.0 * Math.PI / m;
            for (int j = 0; j < quarterM; j++) {
                double angle1 = baseAngle * j;
                double angle2 = baseAngle * 2 * j;
                double angle3 = baseAngle * 3 * j;
                cos1[j] = Math.cos(angle1);
                sin1[j] = Math.sin(angle1);
                cos2[j] = Math.cos(angle2);
                sin2[j] = Math.sin(angle2);
                cos3[j] = Math.cos(angle3);
                sin3[j] = Math.sin(angle3);
            }
            int base = s * 6;
            twiddles[base] = cos1;
            twiddles[base + 1] = sin1;
            twiddles[base + 2] = cos2;
            twiddles[base + 3] = sin2;
            twiddles[base + 4] = cos3;
            twiddles[base + 5] = sin3;
        }
        return twiddles;
    }

    private static float[][] computeRadix4TwiddlesFloat(int n) {
        int numStages = Integer.numberOfTrailingZeros(n) / 2;
        float[][] twiddles = new float[numStages * 6][];

        for (int s = 0; s < numStages; s++) {
            int m = 1 << ((s + 1) * 2);
            int quarterM = m / 4;
            float[] cos1 = new float[quarterM];
            float[] sin1 = new float[quarterM];
            float[] cos2 = new float[quarterM];
            float[] sin2 = new float[quarterM];
            float[] cos3 = new float[quarterM];
            float[] sin3 = new float[quarterM];

            double baseAngle = -2.0 * Math.PI / m;
            for (int j = 0; j < quarterM; j++) {
                cos1[j] = (float) Math.cos(baseAngle * j);
                sin1[j] = (float) Math.sin(baseAngle * j);
                cos2[j] = (float) Math.cos(baseAngle * 2 * j);
                sin2[j] = (float) Math.sin(baseAngle * 2 * j);
                cos3[j] = (float) Math.cos(baseAngle * 3 * j);
                sin3[j] = (float) Math.sin(baseAngle * 3 * j);
            }
            int base = s * 6;
            twiddles[base] = cos1;
            twiddles[base + 1] = sin1;
            twiddles[base + 2] = cos2;
            twiddles[base + 3] = sin2;
            twiddles[base + 4] = cos3;
            twiddles[base + 5] = sin3;
        }
        return twiddles;
    }

    private static double[][] computeRadix2TwiddlesDouble(int n) {
        int halfN = n / 2;
        double[] cos = new double[halfN];
        double[] sin = new double[halfN];
        double angleStep = -2.0 * Math.PI / n;
        for (int j = 0; j < halfN; j++) {
            cos[j] = Math.cos(angleStep * j);
            sin[j] = Math.sin(angleStep * j);
        }
        return new double[][]{cos, sin};
    }

    private static float[][] computeRadix2TwiddlesFloat(int n) {
        int halfN = n / 2;
        float[] cos = new float[halfN];
        float[] sin = new float[halfN];
        double angleStep = -2.0 * Math.PI / n;
        for (int j = 0; j < halfN; j++) {
            cos[j] = (float) Math.cos(angleStep * j);
            sin[j] = (float) Math.sin(angleStep * j);
        }
        return new float[][]{cos, sin};
    }

    private static double[] getTempRealD(int n) {
        double[] buf = TEMP_REAL_D.get();
        if (buf == null || buf.length < n) {
            buf = new double[n];
            TEMP_REAL_D.set(buf);
        }
        return buf;
    }

    private static double[] getTempImagD(int n) {
        double[] buf = TEMP_IMAG_D.get();
        if (buf == null || buf.length < n) {
            buf = new double[n];
            TEMP_IMAG_D.set(buf);
        }
        return buf;
    }

    private static float[] getTempRealF(int n) {
        float[] buf = TEMP_REAL_F.get();
        if (buf == null || buf.length < n) {
            buf = new float[n];
            TEMP_REAL_F.set(buf);
        }
        return buf;
    }

    private static float[] getTempImagF(int n) {
        float[] buf = TEMP_IMAG_F.get();
        if (buf == null || buf.length < n) {
            buf = new float[n];
            TEMP_IMAG_F.set(buf);
        }
        return buf;
    }

    // ==================== RADIX-4 STOCKHAM FFT (DOUBLE) ====================

    public static void fft(double[] real, double[] imag) {
        int n = real.length;
        if (n != imag.length) throw new IllegalArgumentException("Mismatched arrays");
        if (n <= 1) return;
        if ((n & (n - 1)) != 0) throw new IllegalArgumentException("Length must be power of 2");

        // Handle small sizes directly
        if (n == 2) {
            double r0 = real[0], i0 = imag[0], r1 = real[1], i1 = imag[1];
            real[0] = r0 + r1; imag[0] = i0 + i1;
            real[1] = r0 - r1; imag[1] = i0 - i1;
            return;
        }
        if (n == 4) {
            radix4Butterfly4(real, imag);
            return;
        }

        double[] tempReal = getTempRealD(n);
        double[] tempImag = getTempImagD(n);
        double[] srcReal = real, srcImag = imag;
        double[] dstReal = tempReal, dstImag = tempImag;

        int log2n = Integer.numberOfTrailingZeros(n);
        int numRadix4Stages = log2n / 2;
        boolean hasRadix2Final = (log2n & 1) == 1;

        double[][] r4Twiddles = getRadix4TwiddlesDouble(n);

        // Radix-4 stages
        for (int s = 0; s < numRadix4Stages; s++) {
            int m = 1 << ((s + 1) * 2);  // 4, 16, 64, ...
            int quarterM = m / 4;
            int numBlocks = n / m;

            int base = s * 6;
            double[] cos1 = r4Twiddles[base];
            double[] sin1 = r4Twiddles[base + 1];
            double[] cos2 = r4Twiddles[base + 2];
            double[] sin2 = r4Twiddles[base + 3];
            double[] cos3 = r4Twiddles[base + 4];
            double[] sin3 = r4Twiddles[base + 5];

            stockhamRadix4StageDouble(srcReal, srcImag, dstReal, dstImag, n, quarterM, numBlocks,
                    cos1, sin1, cos2, sin2, cos3, sin3);

            double[] t = srcReal; srcReal = dstReal; dstReal = t;
            t = srcImag; srcImag = dstImag; dstImag = t;
        }

        // Final radix-2 stage if needed
        if (hasRadix2Final) {
            double[][] r2Twiddles = getRadix2TwiddlesDouble(n);
            stockhamRadix2StageDouble(srcReal, srcImag, dstReal, dstImag, n, n / 2, 1, r2Twiddles[0], r2Twiddles[1]);
            double[] t = srcReal; srcReal = dstReal; dstReal = t;
            t = srcImag; srcImag = dstImag; dstImag = t;
        }

        if (srcReal != real) {
            System.arraycopy(srcReal, 0, real, 0, n);
            System.arraycopy(srcImag, 0, imag, 0, n);
        }
    }

    private static void radix4Butterfly4(double[] real, double[] imag) {
        double r0 = real[0], i0 = imag[0];
        double r1 = real[1], i1 = imag[1];
        double r2 = real[2], i2 = imag[2];
        double r3 = real[3], i3 = imag[3];

        double t0r = r0 + r2, t0i = i0 + i2;
        double t1r = r0 - r2, t1i = i0 - i2;
        double t2r = r1 + r3, t2i = i1 + i3;
        double t3r = r1 - r3, t3i = i1 - i3;

        real[0] = t0r + t2r; imag[0] = t0i + t2i;
        real[1] = t1r + t3i; imag[1] = t1i - t3r;
        real[2] = t0r - t2r; imag[2] = t0i - t2i;
        real[3] = t1r - t3i; imag[3] = t1i + t3r;
    }

    private static void stockhamRadix4StageDouble(
            double[] srcReal, double[] srcImag, double[] dstReal, double[] dstImag,
            int n, int quarterM, int numBlocks,
            double[] cos1, double[] sin1, double[] cos2, double[] sin2, double[] cos3, double[] sin3) {

        int vecLen = SPECIES.length();
        int gap = n / 4;

        for (int k = 0; k < numBlocks; k++) {
            int srcBase = k * quarterM;
            int dstBase = k * quarterM * 4;

            int j = 0;
            int loopBound = SPECIES.loopBound(quarterM);

            // Unrolled vectorized loop (2 vectors per iteration)
            for (; j + vecLen * 2 <= loopBound; j += vecLen * 2) {
                // First vector
                processRadix4VectorDouble(srcReal, srcImag, dstReal, dstImag,
                        srcBase, dstBase, gap, quarterM, j,
                        cos1, sin1, cos2, sin2, cos3, sin3);
                // Second vector
                processRadix4VectorDouble(srcReal, srcImag, dstReal, dstImag,
                        srcBase, dstBase, gap, quarterM, j + vecLen,
                        cos1, sin1, cos2, sin2, cos3, sin3);
            }

            // Remaining vectors
            for (; j < loopBound; j += vecLen) {
                processRadix4VectorDouble(srcReal, srcImag, dstReal, dstImag,
                        srcBase, dstBase, gap, quarterM, j,
                        cos1, sin1, cos2, sin2, cos3, sin3);
            }

            // Scalar tail
            for (; j < quarterM; j++) {
                double x0r = srcReal[srcBase + j];
                double x0i = srcImag[srcBase + j];
                double x1r = srcReal[srcBase + j + gap];
                double x1i = srcImag[srcBase + j + gap];
                double x2r = srcReal[srcBase + j + 2 * gap];
                double x2i = srcImag[srcBase + j + 2 * gap];
                double x3r = srcReal[srcBase + j + 3 * gap];
                double x3i = srcImag[srcBase + j + 3 * gap];

                double w1r = cos1[j], w1i = sin1[j];
                double w2r = cos2[j], w2i = sin2[j];
                double w3r = cos3[j], w3i = sin3[j];

                double t1r = w1r * x1r - w1i * x1i, t1i = w1r * x1i + w1i * x1r;
                double t2r = w2r * x2r - w2i * x2i, t2i = w2r * x2i + w2i * x2r;
                double t3r = w3r * x3r - w3i * x3i, t3i = w3r * x3i + w3i * x3r;

                double u0r = x0r + t2r, u0i = x0i + t2i;
                double u1r = x0r - t2r, u1i = x0i - t2i;
                double u2r = t1r + t3r, u2i = t1i + t3i;
                double u3r = t1r - t3r, u3i = t1i - t3i;

                dstReal[dstBase + j] = u0r + u2r;
                dstImag[dstBase + j] = u0i + u2i;
                dstReal[dstBase + j + quarterM] = u1r + u3i;
                dstImag[dstBase + j + quarterM] = u1i - u3r;
                dstReal[dstBase + j + 2 * quarterM] = u0r - u2r;
                dstImag[dstBase + j + 2 * quarterM] = u0i - u2i;
                dstReal[dstBase + j + 3 * quarterM] = u1r - u3i;
                dstImag[dstBase + j + 3 * quarterM] = u1i + u3r;
            }
        }
    }

    private static void processRadix4VectorDouble(
            double[] srcReal, double[] srcImag, double[] dstReal, double[] dstImag,
            int srcBase, int dstBase, int gap, int quarterM, int j,
            double[] cos1, double[] sin1, double[] cos2, double[] sin2, double[] cos3, double[] sin3) {

        DoubleVector x0r = DoubleVector.fromArray(SPECIES, srcReal, srcBase + j);
        DoubleVector x0i = DoubleVector.fromArray(SPECIES, srcImag, srcBase + j);
        DoubleVector x1r = DoubleVector.fromArray(SPECIES, srcReal, srcBase + j + gap);
        DoubleVector x1i = DoubleVector.fromArray(SPECIES, srcImag, srcBase + j + gap);
        DoubleVector x2r = DoubleVector.fromArray(SPECIES, srcReal, srcBase + j + 2 * gap);
        DoubleVector x2i = DoubleVector.fromArray(SPECIES, srcImag, srcBase + j + 2 * gap);
        DoubleVector x3r = DoubleVector.fromArray(SPECIES, srcReal, srcBase + j + 3 * gap);
        DoubleVector x3i = DoubleVector.fromArray(SPECIES, srcImag, srcBase + j + 3 * gap);

        DoubleVector w1r = DoubleVector.fromArray(SPECIES, cos1, j);
        DoubleVector w1i = DoubleVector.fromArray(SPECIES, sin1, j);
        DoubleVector w2r = DoubleVector.fromArray(SPECIES, cos2, j);
        DoubleVector w2i = DoubleVector.fromArray(SPECIES, sin2, j);
        DoubleVector w3r = DoubleVector.fromArray(SPECIES, cos3, j);
        DoubleVector w3i = DoubleVector.fromArray(SPECIES, sin3, j);

        DoubleVector t1r = w1r.fma(x1r, w1i.neg().mul(x1i));
        DoubleVector t1i = w1r.fma(x1i, w1i.mul(x1r));
        DoubleVector t2r = w2r.fma(x2r, w2i.neg().mul(x2i));
        DoubleVector t2i = w2r.fma(x2i, w2i.mul(x2r));
        DoubleVector t3r = w3r.fma(x3r, w3i.neg().mul(x3i));
        DoubleVector t3i = w3r.fma(x3i, w3i.mul(x3r));

        DoubleVector u0r = x0r.add(t2r), u0i = x0i.add(t2i);
        DoubleVector u1r = x0r.sub(t2r), u1i = x0i.sub(t2i);
        DoubleVector u2r = t1r.add(t3r), u2i = t1i.add(t3i);
        DoubleVector u3r = t1r.sub(t3r), u3i = t1i.sub(t3i);

        u0r.add(u2r).intoArray(dstReal, dstBase + j);
        u0i.add(u2i).intoArray(dstImag, dstBase + j);
        u1r.add(u3i).intoArray(dstReal, dstBase + j + quarterM);
        u1i.sub(u3r).intoArray(dstImag, dstBase + j + quarterM);
        u0r.sub(u2r).intoArray(dstReal, dstBase + j + 2 * quarterM);
        u0i.sub(u2i).intoArray(dstImag, dstBase + j + 2 * quarterM);
        u1r.sub(u3i).intoArray(dstReal, dstBase + j + 3 * quarterM);
        u1i.add(u3r).intoArray(dstImag, dstBase + j + 3 * quarterM);
    }

    private static void stockhamRadix2StageDouble(
            double[] srcReal, double[] srcImag, double[] dstReal, double[] dstImag,
            int n, int halfM, int numBlocks, double[] cos, double[] sin) {

        int vecLen = SPECIES.length();

        for (int k = 0; k < numBlocks; k++) {
            int srcBase = k * halfM;
            int dstBase = k * halfM * 2;

            int j = 0;
            int loopBound = SPECIES.loopBound(halfM);

            // Unrolled loop (2 vectors)
            for (; j + vecLen * 2 <= loopBound; j += vecLen * 2) {
                processRadix2VectorDouble(srcReal, srcImag, dstReal, dstImag, srcBase, dstBase, n, halfM, j, cos, sin);
                processRadix2VectorDouble(srcReal, srcImag, dstReal, dstImag, srcBase, dstBase, n, halfM, j + vecLen, cos, sin);
            }
            for (; j < loopBound; j += vecLen) {
                processRadix2VectorDouble(srcReal, srcImag, dstReal, dstImag, srcBase, dstBase, n, halfM, j, cos, sin);
            }

            for (; j < halfM; j++) {
                double ar = srcReal[srcBase + j], ai = srcImag[srcBase + j];
                double br = srcReal[srcBase + j + n / 2], bi = srcImag[srcBase + j + n / 2];
                double wr = cos[j], wi = sin[j];
                double tr = wr * br - wi * bi, ti = wr * bi + wi * br;
                dstReal[dstBase + j] = ar + tr; dstImag[dstBase + j] = ai + ti;
                dstReal[dstBase + j + halfM] = ar - tr; dstImag[dstBase + j + halfM] = ai - ti;
            }
        }
    }

    private static void processRadix2VectorDouble(
            double[] srcReal, double[] srcImag, double[] dstReal, double[] dstImag,
            int srcBase, int dstBase, int n, int halfM, int j, double[] cos, double[] sin) {
        DoubleVector ar = DoubleVector.fromArray(SPECIES, srcReal, srcBase + j);
        DoubleVector ai = DoubleVector.fromArray(SPECIES, srcImag, srcBase + j);
        DoubleVector br = DoubleVector.fromArray(SPECIES, srcReal, srcBase + j + n / 2);
        DoubleVector bi = DoubleVector.fromArray(SPECIES, srcImag, srcBase + j + n / 2);
        DoubleVector wr = DoubleVector.fromArray(SPECIES, cos, j);
        DoubleVector wi = DoubleVector.fromArray(SPECIES, sin, j);
        DoubleVector tr = wr.fma(br, wi.neg().mul(bi));
        DoubleVector ti = wr.fma(bi, wi.mul(br));
        ar.add(tr).intoArray(dstReal, dstBase + j);
        ai.add(ti).intoArray(dstImag, dstBase + j);
        ar.sub(tr).intoArray(dstReal, dstBase + j + halfM);
        ai.sub(ti).intoArray(dstImag, dstBase + j + halfM);
    }

    // ==================== RADIX-4 STOCKHAM FFT (FLOAT) ====================

    public static void fft(float[] real, float[] imag) {
        int n = real.length;
        if (n != imag.length) throw new IllegalArgumentException("Mismatched arrays");
        if (n <= 1) return;
        if ((n & (n - 1)) != 0) throw new IllegalArgumentException("Length must be power of 2");

        if (n == 2) {
            float r0 = real[0], i0 = imag[0], r1 = real[1], i1 = imag[1];
            real[0] = r0 + r1; imag[0] = i0 + i1;
            real[1] = r0 - r1; imag[1] = i0 - i1;
            return;
        }
        if (n == 4) {
            radix4Butterfly4(real, imag);
            return;
        }

        float[] tempReal = getTempRealF(n);
        float[] tempImag = getTempImagF(n);
        float[] srcReal = real, srcImag = imag;
        float[] dstReal = tempReal, dstImag = tempImag;

        int log2n = Integer.numberOfTrailingZeros(n);
        int numRadix4Stages = log2n / 2;
        boolean hasRadix2Final = (log2n & 1) == 1;

        float[][] r4Twiddles = getRadix4TwiddlesFloat(n);

        for (int s = 0; s < numRadix4Stages; s++) {
            int m = 1 << ((s + 1) * 2);
            int quarterM = m / 4;
            int numBlocks = n / m;

            int base = s * 6;
            stockhamRadix4StageFloat(srcReal, srcImag, dstReal, dstImag, n, quarterM, numBlocks,
                    r4Twiddles[base], r4Twiddles[base + 1], r4Twiddles[base + 2],
                    r4Twiddles[base + 3], r4Twiddles[base + 4], r4Twiddles[base + 5]);

            float[] t = srcReal; srcReal = dstReal; dstReal = t;
            t = srcImag; srcImag = dstImag; dstImag = t;
        }

        if (hasRadix2Final) {
            float[][] r2Twiddles = getRadix2TwiddlesFloat(n);
            stockhamRadix2StageFloat(srcReal, srcImag, dstReal, dstImag, n, n / 2, 1, r2Twiddles[0], r2Twiddles[1]);
            float[] t = srcReal; srcReal = dstReal; dstReal = t;
            t = srcImag; srcImag = dstImag; dstImag = t;
        }

        if (srcReal != real) {
            System.arraycopy(srcReal, 0, real, 0, n);
            System.arraycopy(srcImag, 0, imag, 0, n);
        }
    }

    private static void radix4Butterfly4(float[] real, float[] imag) {
        float r0 = real[0], i0 = imag[0];
        float r1 = real[1], i1 = imag[1];
        float r2 = real[2], i2 = imag[2];
        float r3 = real[3], i3 = imag[3];

        float t0r = r0 + r2, t0i = i0 + i2;
        float t1r = r0 - r2, t1i = i0 - i2;
        float t2r = r1 + r3, t2i = i1 + i3;
        float t3r = r1 - r3, t3i = i1 - i3;

        real[0] = t0r + t2r; imag[0] = t0i + t2i;
        real[1] = t1r + t3i; imag[1] = t1i - t3r;
        real[2] = t0r - t2r; imag[2] = t0i - t2i;
        real[3] = t1r - t3i; imag[3] = t1i + t3r;
    }

    private static void stockhamRadix4StageFloat(
            float[] srcReal, float[] srcImag, float[] dstReal, float[] dstImag,
            int n, int quarterM, int numBlocks,
            float[] cos1, float[] sin1, float[] cos2, float[] sin2, float[] cos3, float[] sin3) {

        int vecLen = SPECIES_FLOAT.length();
        int gap = n / 4;

        for (int k = 0; k < numBlocks; k++) {
            int srcBase = k * quarterM;
            int dstBase = k * quarterM * 4;

            int j = 0;
            int loopBound = SPECIES_FLOAT.loopBound(quarterM);

            for (; j + vecLen * 2 <= loopBound; j += vecLen * 2) {
                processRadix4VectorFloat(srcReal, srcImag, dstReal, dstImag,
                        srcBase, dstBase, gap, quarterM, j, cos1, sin1, cos2, sin2, cos3, sin3);
                processRadix4VectorFloat(srcReal, srcImag, dstReal, dstImag,
                        srcBase, dstBase, gap, quarterM, j + vecLen, cos1, sin1, cos2, sin2, cos3, sin3);
            }
            for (; j < loopBound; j += vecLen) {
                processRadix4VectorFloat(srcReal, srcImag, dstReal, dstImag,
                        srcBase, dstBase, gap, quarterM, j, cos1, sin1, cos2, sin2, cos3, sin3);
            }

            for (; j < quarterM; j++) {
                float x0r = srcReal[srcBase + j], x0i = srcImag[srcBase + j];
                float x1r = srcReal[srcBase + j + gap], x1i = srcImag[srcBase + j + gap];
                float x2r = srcReal[srcBase + j + 2 * gap], x2i = srcImag[srcBase + j + 2 * gap];
                float x3r = srcReal[srcBase + j + 3 * gap], x3i = srcImag[srcBase + j + 3 * gap];

                float w1r = cos1[j], w1i = sin1[j];
                float w2r = cos2[j], w2i = sin2[j];
                float w3r = cos3[j], w3i = sin3[j];

                float t1r = w1r * x1r - w1i * x1i, t1i = w1r * x1i + w1i * x1r;
                float t2r = w2r * x2r - w2i * x2i, t2i = w2r * x2i + w2i * x2r;
                float t3r = w3r * x3r - w3i * x3i, t3i = w3r * x3i + w3i * x3r;

                float u0r = x0r + t2r, u0i = x0i + t2i;
                float u1r = x0r - t2r, u1i = x0i - t2i;
                float u2r = t1r + t3r, u2i = t1i + t3i;
                float u3r = t1r - t3r, u3i = t1i - t3i;

                dstReal[dstBase + j] = u0r + u2r; dstImag[dstBase + j] = u0i + u2i;
                dstReal[dstBase + j + quarterM] = u1r + u3i; dstImag[dstBase + j + quarterM] = u1i - u3r;
                dstReal[dstBase + j + 2 * quarterM] = u0r - u2r; dstImag[dstBase + j + 2 * quarterM] = u0i - u2i;
                dstReal[dstBase + j + 3 * quarterM] = u1r - u3i; dstImag[dstBase + j + 3 * quarterM] = u1i + u3r;
            }
        }
    }

    private static void processRadix4VectorFloat(
            float[] srcReal, float[] srcImag, float[] dstReal, float[] dstImag,
            int srcBase, int dstBase, int gap, int quarterM, int j,
            float[] cos1, float[] sin1, float[] cos2, float[] sin2, float[] cos3, float[] sin3) {

        FloatVector x0r = FloatVector.fromArray(SPECIES_FLOAT, srcReal, srcBase + j);
        FloatVector x0i = FloatVector.fromArray(SPECIES_FLOAT, srcImag, srcBase + j);
        FloatVector x1r = FloatVector.fromArray(SPECIES_FLOAT, srcReal, srcBase + j + gap);
        FloatVector x1i = FloatVector.fromArray(SPECIES_FLOAT, srcImag, srcBase + j + gap);
        FloatVector x2r = FloatVector.fromArray(SPECIES_FLOAT, srcReal, srcBase + j + 2 * gap);
        FloatVector x2i = FloatVector.fromArray(SPECIES_FLOAT, srcImag, srcBase + j + 2 * gap);
        FloatVector x3r = FloatVector.fromArray(SPECIES_FLOAT, srcReal, srcBase + j + 3 * gap);
        FloatVector x3i = FloatVector.fromArray(SPECIES_FLOAT, srcImag, srcBase + j + 3 * gap);

        FloatVector w1r = FloatVector.fromArray(SPECIES_FLOAT, cos1, j);
        FloatVector w1i = FloatVector.fromArray(SPECIES_FLOAT, sin1, j);
        FloatVector w2r = FloatVector.fromArray(SPECIES_FLOAT, cos2, j);
        FloatVector w2i = FloatVector.fromArray(SPECIES_FLOAT, sin2, j);
        FloatVector w3r = FloatVector.fromArray(SPECIES_FLOAT, cos3, j);
        FloatVector w3i = FloatVector.fromArray(SPECIES_FLOAT, sin3, j);

        FloatVector t1r = w1r.fma(x1r, w1i.neg().mul(x1i));
        FloatVector t1i = w1r.fma(x1i, w1i.mul(x1r));
        FloatVector t2r = w2r.fma(x2r, w2i.neg().mul(x2i));
        FloatVector t2i = w2r.fma(x2i, w2i.mul(x2r));
        FloatVector t3r = w3r.fma(x3r, w3i.neg().mul(x3i));
        FloatVector t3i = w3r.fma(x3i, w3i.mul(x3r));

        FloatVector u0r = x0r.add(t2r), u0i = x0i.add(t2i);
        FloatVector u1r = x0r.sub(t2r), u1i = x0i.sub(t2i);
        FloatVector u2r = t1r.add(t3r), u2i = t1i.add(t3i);
        FloatVector u3r = t1r.sub(t3r), u3i = t1i.sub(t3i);

        u0r.add(u2r).intoArray(dstReal, dstBase + j);
        u0i.add(u2i).intoArray(dstImag, dstBase + j);
        u1r.add(u3i).intoArray(dstReal, dstBase + j + quarterM);
        u1i.sub(u3r).intoArray(dstImag, dstBase + j + quarterM);
        u0r.sub(u2r).intoArray(dstReal, dstBase + j + 2 * quarterM);
        u0i.sub(u2i).intoArray(dstImag, dstBase + j + 2 * quarterM);
        u1r.sub(u3i).intoArray(dstReal, dstBase + j + 3 * quarterM);
        u1i.add(u3r).intoArray(dstImag, dstBase + j + 3 * quarterM);
    }

    private static void stockhamRadix2StageFloat(
            float[] srcReal, float[] srcImag, float[] dstReal, float[] dstImag,
            int n, int halfM, int numBlocks, float[] cos, float[] sin) {

        int vecLen = SPECIES_FLOAT.length();

        for (int k = 0; k < numBlocks; k++) {
            int srcBase = k * halfM;
            int dstBase = k * halfM * 2;

            int j = 0;
            int loopBound = SPECIES_FLOAT.loopBound(halfM);

            for (; j + vecLen * 2 <= loopBound; j += vecLen * 2) {
                processRadix2VectorFloat(srcReal, srcImag, dstReal, dstImag, srcBase, dstBase, n, halfM, j, cos, sin);
                processRadix2VectorFloat(srcReal, srcImag, dstReal, dstImag, srcBase, dstBase, n, halfM, j + vecLen, cos, sin);
            }
            for (; j < loopBound; j += vecLen) {
                processRadix2VectorFloat(srcReal, srcImag, dstReal, dstImag, srcBase, dstBase, n, halfM, j, cos, sin);
            }

            for (; j < halfM; j++) {
                float ar = srcReal[srcBase + j], ai = srcImag[srcBase + j];
                float br = srcReal[srcBase + j + n / 2], bi = srcImag[srcBase + j + n / 2];
                float wr = cos[j], wi = sin[j];
                float tr = wr * br - wi * bi, ti = wr * bi + wi * br;
                dstReal[dstBase + j] = ar + tr; dstImag[dstBase + j] = ai + ti;
                dstReal[dstBase + j + halfM] = ar - tr; dstImag[dstBase + j + halfM] = ai - ti;
            }
        }
    }

    private static void processRadix2VectorFloat(
            float[] srcReal, float[] srcImag, float[] dstReal, float[] dstImag,
            int srcBase, int dstBase, int n, int halfM, int j, float[] cos, float[] sin) {
        FloatVector ar = FloatVector.fromArray(SPECIES_FLOAT, srcReal, srcBase + j);
        FloatVector ai = FloatVector.fromArray(SPECIES_FLOAT, srcImag, srcBase + j);
        FloatVector br = FloatVector.fromArray(SPECIES_FLOAT, srcReal, srcBase + j + n / 2);
        FloatVector bi = FloatVector.fromArray(SPECIES_FLOAT, srcImag, srcBase + j + n / 2);
        FloatVector wr = FloatVector.fromArray(SPECIES_FLOAT, cos, j);
        FloatVector wi = FloatVector.fromArray(SPECIES_FLOAT, sin, j);
        FloatVector tr = wr.fma(br, wi.neg().mul(bi));
        FloatVector ti = wr.fma(bi, wi.mul(br));
        ar.add(tr).intoArray(dstReal, dstBase + j);
        ai.add(ti).intoArray(dstImag, dstBase + j);
        ar.sub(tr).intoArray(dstReal, dstBase + j + halfM);
        ai.sub(ti).intoArray(dstImag, dstBase + j + halfM);
    }

    // ==================== UTILITY ====================

    public static void clearCache() {
        TWIDDLE_CACHE_R4_D.clear();
        TWIDDLE_CACHE_R4_F.clear();
        TWIDDLE_CACHE_R2_D.clear();
        TWIDDLE_CACHE_R2_F.clear();
        UNPACK_TWIDDLE_D.clear();
        UNPACK_TWIDDLE_F.clear();
    }

    // ==================== OPTIMIZED fftReal ====================

    private static double[][] getUnpackTwiddlesDouble(int halfN) {
        return UNPACK_TWIDDLE_D.computeIfAbsent(halfN, k -> {
            double[] cos = new double[k];
            double[] sin = new double[k];
            for (int j = 0; j < k; j++) {
                double angle = -Math.PI * j / k;
                cos[j] = Math.cos(angle);
                sin[j] = Math.sin(angle);
            }
            return new double[][]{cos, sin};
        });
    }

    private static float[][] getUnpackTwiddlesFloat(int halfN) {
        return UNPACK_TWIDDLE_F.computeIfAbsent(halfN, k -> {
            float[] cos = new float[k];
            float[] sin = new float[k];
            for (int j = 0; j < k; j++) {
                double angle = -Math.PI * j / k;
                cos[j] = (float) Math.cos(angle);
                sin[j] = (float) Math.sin(angle);
            }
            return new float[][]{cos, sin};
        });
    }

    public static void fftReal(float[] input, float[] outReal, float[] outImag) {
        int n = input.length;
        if ((n & (n - 1)) != 0) throw new IllegalArgumentException("Length must be power of 2");
        int halfN = n / 2;

        // Must allocate new arrays - can't reuse temp pool since fft() uses them
        float[] zReal = new float[halfN];
        float[] zImag = new float[halfN];

        // Pack input into complex array
        for (int j = 0; j < halfN; j++) {
            zReal[j] = input[2 * j];
            zImag[j] = input[2 * j + 1];
        }

        // Use optimized Stockham FFT
        fft(zReal, zImag);

        // Unpack
        outReal[0] = zReal[0] + zImag[0];
        outImag[0] = 0;
        outReal[halfN] = zReal[0] - zImag[0];
        outImag[halfN] = 0;

        float[][] unpackTw = getUnpackTwiddlesFloat(halfN);
        float[] twCos = unpackTw[0];
        float[] twSin = unpackTw[1];

        int vecLen = SPECIES_FLOAT.length();
        FloatVector half = FloatVector.broadcast(SPECIES_FLOAT, 0.5f);
        int j = 1;
        int loopBound = SPECIES_FLOAT.loopBound(halfN - 1);

        for (; j < loopBound + 1; j += vecLen) {
            if (j + vecLen > halfN) break;

            FloatVector zkR = FloatVector.fromArray(SPECIES_FLOAT, zReal, j);
            FloatVector zkI = FloatVector.fromArray(SPECIES_FLOAT, zImag, j);

            // Load conjugate symmetry values (reversed)
            float[] zcRArr = new float[vecLen];
            float[] zcIArr = new float[vecLen];
            for (int k = 0; k < vecLen; k++) {
                int idx = halfN - (j + k);
                zcRArr[k] = zReal[idx];
                zcIArr[k] = -zImag[idx];
            }
            FloatVector zcR = FloatVector.fromArray(SPECIES_FLOAT, zcRArr, 0);
            FloatVector zcI = FloatVector.fromArray(SPECIES_FLOAT, zcIArr, 0);

            FloatVector aR = zkR.add(zcR), aI = zkI.add(zcI);
            FloatVector bR = zkR.sub(zcR), bI = zkI.sub(zcI);
            FloatVector ibR = bI.neg(), ibI = bR;

            FloatVector wR = FloatVector.fromArray(SPECIES_FLOAT, twCos, j);
            FloatVector wI = FloatVector.fromArray(SPECIES_FLOAT, twSin, j);
            FloatVector wibR = wR.mul(ibR).sub(wI.mul(ibI));
            FloatVector wibI = wR.mul(ibI).add(wI.mul(ibR));

            aR.sub(wibR).mul(half).intoArray(outReal, j);
            aI.sub(wibI).mul(half).intoArray(outImag, j);
        }

        for (; j < halfN; j++) {
            int conj = halfN - j;
            float zkR = zReal[j], zkI = zImag[j];
            float zcR = zReal[conj], zcI = -zImag[conj];
            float aR = zkR + zcR, aI = zkI + zcI;
            float bR = zkR - zcR, bI = zkI - zcI;
            float ibR = -bI, ibI = bR;
            float wR = twCos[j], wI = twSin[j];
            float wibR = wR * ibR - wI * ibI, wibI = wR * ibI + wI * ibR;
            outReal[j] = 0.5f * (aR - wibR);
            outImag[j] = 0.5f * (aI - wibI);
        }
    }

    public static void fftReal(double[] input, double[] outReal, double[] outImag) {
        int n = input.length;
        if ((n & (n - 1)) != 0) throw new IllegalArgumentException("Length must be power of 2");
        int halfN = n / 2;

        // Must allocate new arrays - can't reuse temp pool since fft() uses them
        double[] zReal = new double[halfN];
        double[] zImag = new double[halfN];

        for (int j = 0; j < halfN; j++) {
            zReal[j] = input[2 * j];
            zImag[j] = input[2 * j + 1];
        }

        fft(zReal, zImag);

        outReal[0] = zReal[0] + zImag[0];
        outImag[0] = 0;
        outReal[halfN] = zReal[0] - zImag[0];
        outImag[halfN] = 0;

        double[][] unpackTw = getUnpackTwiddlesDouble(halfN);
        double[] twCos = unpackTw[0];
        double[] twSin = unpackTw[1];

        int vecLen = SPECIES.length();
        DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);
        int j = 1;
        int loopBound = SPECIES.loopBound(halfN - 1);

        for (; j < loopBound + 1; j += vecLen) {
            if (j + vecLen > halfN) break;

            DoubleVector zkR = DoubleVector.fromArray(SPECIES, zReal, j);
            DoubleVector zkI = DoubleVector.fromArray(SPECIES, zImag, j);

            double[] zcRArr = new double[vecLen];
            double[] zcIArr = new double[vecLen];
            for (int k = 0; k < vecLen; k++) {
                int idx = halfN - (j + k);
                zcRArr[k] = zReal[idx];
                zcIArr[k] = -zImag[idx];
            }
            DoubleVector zcR = DoubleVector.fromArray(SPECIES, zcRArr, 0);
            DoubleVector zcI = DoubleVector.fromArray(SPECIES, zcIArr, 0);

            DoubleVector aR = zkR.add(zcR), aI = zkI.add(zcI);
            DoubleVector bR = zkR.sub(zcR), bI = zkI.sub(zcI);
            DoubleVector ibR = bI.neg(), ibI = bR;

            DoubleVector wR = DoubleVector.fromArray(SPECIES, twCos, j);
            DoubleVector wI = DoubleVector.fromArray(SPECIES, twSin, j);
            DoubleVector wibR = wR.mul(ibR).sub(wI.mul(ibI));
            DoubleVector wibI = wR.mul(ibI).add(wI.mul(ibR));

            aR.sub(wibR).mul(half).intoArray(outReal, j);
            aI.sub(wibI).mul(half).intoArray(outImag, j);
        }

        for (; j < halfN; j++) {
            int conj = halfN - j;
            double zkR = zReal[j], zkI = zImag[j];
            double zcR = zReal[conj], zcI = -zImag[conj];
            double aR = zkR + zcR, aI = zkI + zcI;
            double bR = zkR - zcR, bI = zkI - zcI;
            double ibR = -bI, ibI = bR;
            double wR = twCos[j], wI = twSin[j];
            double wibR = wR * ibR - wI * ibI, wibI = wR * ibI + wI * ibR;
            outReal[j] = 0.5 * (aR - wibR);
            outImag[j] = 0.5 * (aI - wibI);
        }
    }

    static void main() {
        int N = 1024; // Must be power of 2

        // Double precision FFT
        double[] realD = new double[N];
        double[] imagD = new double[N];
        for (int i = 0; i < N / 2; i++) realD[i] = 1.0;

        IO.print("Processing double FFT with Vector API (Species: " + SPECIES + ")...");
        fft(realD, imagD);
        IO.print("Double Result[0]: " + String.format("%.2f + %.2fi", realD[0], imagD[0]));
        IO.print("Double Result[1]: " + String.format("%.2f + %.2fi", realD[1], imagD[1]));

        // Single precision (float) FFT
        float[] realF = new float[N];
        float[] imagF = new float[N];
        for (int i = 0; i < N / 2; i++) realF[i] = 1.0f;

        IO.print("\nProcessing float FFT with Vector API (Species: " + SPECIES_FLOAT + ")...");
        fft(realF, imagF);
        IO.print("Float Result[0]: " + String.format("%.2f + %.2fi", realF[0], imagF[0]));
        IO.print("Float Result[1]: " + String.format("%.2f + %.2fi", realF[1], imagF[1]));

        // Real-only FFT (float) - optimized for real signals
        float[] realInput = new float[N];
        for (int i = 0; i < N / 2; i++) realInput[i] = 1.0f;
        float[] outRealF = new float[N / 2 + 1];
        float[] outImagF = new float[N / 2 + 1];

        IO.print("\nProcessing real-only float FFT (optimized)...");
        fftReal(realInput, outRealF, outImagF);
        IO.print("Real FFT Result[0]: " + String.format("%.2f + %.2fi", outRealF[0], outImagF[0]));
        IO.print("Real FFT Result[1]: " + String.format("%.2f + %.2fi", outRealF[1], outImagF[1]));

        // Real-only FFT (double)
        double[] realInputD = new double[N];
        for (int i = 0; i < N / 2; i++) realInputD[i] = 1.0;
        double[] outRealD = new double[N / 2 + 1];
        double[] outImagD = new double[N / 2 + 1];

        IO.print("\nProcessing real-only double FFT (optimized)...");
        fftReal(realInputD, outRealD, outImagD);
        IO.print("Real FFT Result[0]: " + String.format("%.2f + %.2fi", outRealD[0], outImagD[0]));
        IO.print("Real FFT Result[1]: " + String.format("%.2f + %.2fi", outRealD[1], outImagD[1]));
    }
}