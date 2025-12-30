package org.example;

import jdk.incubator.vector.*;

public class VectorFFT {

    // Select the best vector size for the current CPU (e.g., 256-bit or 512-bit)
    static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    static final VectorSpecies<Float> SPECIES_FLOAT = FloatVector.SPECIES_PREFERRED;

    public static void fft(double[] real, double[] imag) {
        int n = real.length;
        if (n != imag.length) throw new IllegalArgumentException("Mismatched arrays");

        // 1. Bit-Reversal Permutation (Scalar is fast enough here)
        int shift = 1 + Integer.numberOfLeadingZeros(n);
        for (int k = 0; k < n; k++) {
            int j = Integer.reverse(k) >>> shift;
            if (j > k) {
                double tr = real[j]; real[j] = real[k]; real[k] = tr;
                double ti = imag[j]; imag[j] = imag[k]; imag[k] = ti;
            }
        }

        // 2. Precompute Twiddle Factors (Cos/Sin tables)
        // This avoids calculating Math.cos/sin inside the critical loop
        double[] cosTable = new double[n / 2];
        double[] sinTable = new double[n / 2];
        for (int i = 0; i < n / 2; i++) {
            double angle = -2 * Math.PI * i / n;
            cosTable[i] = Math.cos(angle);
            sinTable[i] = Math.sin(angle);
        }

        // 3. Butterfly Operations
        for (int len = 2; len <= n; len <<= 1) {
            int halfLen = len / 2;
            int step = n / len; // Step size for accessing precomputed tables

            // Process blocks of size 'len'
            for (int i = 0; i < n; i += len) {

                int j = 0;

                // --- VECTOR LOOP ---
                // Process 'SPECIES.length()' elements (e.g., 4 or 8) at a time
                int loopBound = SPECIES.loopBound(halfLen);
                for (; j < loopBound; j += SPECIES.length()) {
                    int u = i + j;
                    int v = i + j + halfLen;

                    // Load Data Vectors (Real and Imaginary parts)
                    DoubleVector u_re = DoubleVector.fromArray(SPECIES, real, u);
                    DoubleVector u_im = DoubleVector.fromArray(SPECIES, imag, u);
                    DoubleVector v_re = DoubleVector.fromArray(SPECIES, real, v);
                    DoubleVector v_im = DoubleVector.fromArray(SPECIES, imag, v);

                    // Gather Twiddle Factors
                    // Since twiddle factors are spaced by 'step', we must gather them manually
                    // or (if strictly optimizing) pre-arrange tables for each stage.
                    // For simplicity here, we use a gather or linear load if contiguous.
                    // Note: In standard FFT, twiddles for a block are NOT contiguous in the global table
                    // unless we permute them. To keep this readable, we load using a gathered index.

                    // Load twiddle factors for this vector iteration
                    // Note: Twiddle factors are spaced by 'step' in the precomputed table
                    double[] w_re_arr = new double[SPECIES.length()];
                    double[] w_im_arr = new double[SPECIES.length()];
                    for(int k=0; k<SPECIES.length(); k++) {
                        int idx = (j + k) * step;
                        w_re_arr[k] = cosTable[idx];
                        w_im_arr[k] = sinTable[idx];
                    }
                    DoubleVector w_re = DoubleVector.fromArray(SPECIES, w_re_arr, 0);
                    DoubleVector w_im = DoubleVector.fromArray(SPECIES, w_im_arr, 0);

                    // Complex Multiply: w * v
                    // (wr*vr - wi*vi) + i(wr*vi + wi*vr)
                    DoubleVector t_re = w_re.mul(v_re).sub(w_im.mul(v_im));
                    DoubleVector t_im = w_re.mul(v_im).add(w_im.mul(v_re));

                    // Butterfly Operation
                    // u = u + t
                    // v = u - t
                    u_re.add(t_re).intoArray(real, u);
                    u_im.add(t_im).intoArray(imag, u);

                    u_re.sub(t_re).intoArray(real, v);
                    u_im.sub(t_im).intoArray(imag, v);
                }

                // --- SCALAR TAIL LOOP ---
                // Handle remaining elements that don't fit in a full vector
                for (; j < halfLen; j++) {
                    int u = i + j;
                    int v = i + j + halfLen;

                    // Load Twiddle directly from table
                    double wr = cosTable[j * step];
                    double wi = sinTable[j * step];

                    double vr = real[v];
                    double vi = imag[v];

                    double tr = wr * vr - wi * vi;
                    double ti = wr * vi + wi * vr;

                    double ur = real[u];
                    double ui = imag[u];

                    real[u] = ur + tr;
                    imag[u] = ui + ti;
                    real[v] = ur - tr;
                    imag[v] = ui - ti;
                }
            }
        }
    }

    public static void fft(float[] real, float[] imag) {
        int n = real.length;
        if (n != imag.length) throw new IllegalArgumentException("Mismatched arrays");

        // 1. Bit-Reversal Permutation (Scalar is fast enough here)
        int shift = 1 + Integer.numberOfLeadingZeros(n);
        for (int k = 0; k < n; k++) {
            int j = Integer.reverse(k) >>> shift;
            if (j > k) {
                float tr = real[j]; real[j] = real[k]; real[k] = tr;
                float ti = imag[j]; imag[j] = imag[k]; imag[k] = ti;
            }
        }

        // 2. Precompute Twiddle Factors (Cos/Sin tables)
        float[] cosTable = new float[n / 2];
        float[] sinTable = new float[n / 2];
        for (int i = 0; i < n / 2; i++) {
            double angle = -2 * Math.PI * i / n;
            cosTable[i] = (float) Math.cos(angle);
            sinTable[i] = (float) Math.sin(angle);
        }

        // 3. Butterfly Operations
        for (int len = 2; len <= n; len <<= 1) {
            int halfLen = len / 2;
            int step = n / len;

            for (int i = 0; i < n; i += len) {

                int j = 0;

                // --- VECTOR LOOP ---
                int loopBound = SPECIES_FLOAT.loopBound(halfLen);
                for (; j < loopBound; j += SPECIES_FLOAT.length()) {
                    int u = i + j;
                    int v = i + j + halfLen;

                    // Load Data Vectors (Real and Imaginary parts)
                    FloatVector u_re = FloatVector.fromArray(SPECIES_FLOAT, real, u);
                    FloatVector u_im = FloatVector.fromArray(SPECIES_FLOAT, imag, u);
                    FloatVector v_re = FloatVector.fromArray(SPECIES_FLOAT, real, v);
                    FloatVector v_im = FloatVector.fromArray(SPECIES_FLOAT, imag, v);

                    // Load twiddle factors for this vector iteration
                    float[] w_re_arr = new float[SPECIES_FLOAT.length()];
                    float[] w_im_arr = new float[SPECIES_FLOAT.length()];
                    for (int k = 0; k < SPECIES_FLOAT.length(); k++) {
                        int idx = (j + k) * step;
                        w_re_arr[k] = cosTable[idx];
                        w_im_arr[k] = sinTable[idx];
                    }
                    FloatVector w_re = FloatVector.fromArray(SPECIES_FLOAT, w_re_arr, 0);
                    FloatVector w_im = FloatVector.fromArray(SPECIES_FLOAT, w_im_arr, 0);

                    // Complex Multiply: w * v
                    FloatVector t_re = w_re.mul(v_re).sub(w_im.mul(v_im));
                    FloatVector t_im = w_re.mul(v_im).add(w_im.mul(v_re));

                    // Butterfly Operation
                    u_re.add(t_re).intoArray(real, u);
                    u_im.add(t_im).intoArray(imag, u);

                    u_re.sub(t_re).intoArray(real, v);
                    u_im.sub(t_im).intoArray(imag, v);
                }

                // --- SCALAR TAIL LOOP ---
                for (; j < halfLen; j++) {
                    int u = i + j;
                    int v = i + j + halfLen;

                    float wr = cosTable[j * step];
                    float wi = sinTable[j * step];

                    float vr = real[v];
                    float vi = imag[v];

                    float tr = wr * vr - wi * vi;
                    float ti = wr * vi + wi * vr;

                    float ur = real[u];
                    float ui = imag[u];

                    real[u] = ur + tr;
                    imag[u] = ui + ti;
                    real[v] = ur - tr;
                    imag[v] = ui - ti;
                }
            }
        }
    }

    /**
     * Optimized FFT for real-valued input signals (float).
     * Uses the "pack and unpack" method: treats N real values as N/2 complex values,
     * performs N/2 complex FFT, then unpacks using conjugate symmetry.
     *
     * @param input   Real input signal of length N (must be power of 2)
     * @param outReal Output: real part of spectrum (length N/2 + 1 for unique values)
     * @param outImag Output: imaginary part of spectrum (length N/2 + 1 for unique values)
     */
    public static void fftReal(float[] input, float[] outReal, float[] outImag) {
        int n = input.length;
        if ((n & (n - 1)) != 0) throw new IllegalArgumentException("Length must be power of 2");

        int halfN = n / 2;

        // Step 1: Pack N real values into N/2 complex values
        // z[k] = x[2k] + i*x[2k+1]
        float[] zReal = new float[halfN];
        float[] zImag = new float[halfN];

        // Scalar packing (deinterleaving is complex with current Vector API)
        for (int j = 0; j < halfN; j++) {
            zReal[j] = input[2 * j];
            zImag[j] = input[2 * j + 1];
        }

        // Step 2: Perform N/2 complex FFT (uses Vector API internally)
        fft(zReal, zImag);

        // Step 3: Unpack to get N-point real FFT
        // Using: X[k] = 0.5 * [(Z[k] + Z*[N/2-k]) - i*W^k*(Z[k] - Z*[N/2-k])]

        // DC component (k=0)
        outReal[0] = zReal[0] + zImag[0];
        outImag[0] = 0;

        // Nyquist component (k=N/2)
        outReal[halfN] = zReal[0] - zImag[0];
        outImag[halfN] = 0;

        // Precompute twiddle factors and reversed conjugate arrays for vectorization
        float[] twReal = new float[halfN];
        float[] twImag = new float[halfN];
        float[] zRealRev = new float[halfN];  // zReal reversed
        float[] zImagRev = new float[halfN];  // zImag reversed (negated for conjugate)
        for (int k = 0; k < halfN; k++) {
            double angle = -Math.PI * k / halfN;
            twReal[k] = (float) Math.cos(angle);
            twImag[k] = (float) Math.sin(angle);
            zRealRev[k] = zReal[halfN - 1 - k];
            zImagRev[k] = -zImag[halfN - 1 - k];  // conjugate
        }

        // Vectorized unpacking (k = 1 to N/2-1)
        // Note: zRealRev[k-1] = zReal[halfN - k], zImagRev[k-1] = -zImag[halfN - k]
        int j = 1;
        int loopBound = SPECIES_FLOAT.loopBound(halfN - 1);
        FloatVector half = FloatVector.broadcast(SPECIES_FLOAT, 0.5f);

        for (; j < loopBound + 1; j += SPECIES_FLOAT.length()) {
            if (j + SPECIES_FLOAT.length() > halfN) break;

            // Load Z[k]
            FloatVector zkR = FloatVector.fromArray(SPECIES_FLOAT, zReal, j);
            FloatVector zkI = FloatVector.fromArray(SPECIES_FLOAT, zImag, j);

            // Load Z*[N/2-k] from pre-reversed arrays (index j-1 maps to halfN-j)
            FloatVector zcR = FloatVector.fromArray(SPECIES_FLOAT, zRealRev, j - 1);
            FloatVector zcI = FloatVector.fromArray(SPECIES_FLOAT, zImagRev, j - 1);

            // A = Z[k] + Z*[N/2-k]
            FloatVector aR = zkR.add(zcR);
            FloatVector aI = zkI.add(zcI);

            // B = Z[k] - Z*[N/2-k]
            FloatVector bR = zkR.sub(zcR);
            FloatVector bI = zkI.sub(zcI);

            // i*B = (-bI, bR)
            FloatVector ibR = bI.neg();
            FloatVector ibI = bR;

            // Load twiddle factors W^k
            FloatVector wR = FloatVector.fromArray(SPECIES_FLOAT, twReal, j);
            FloatVector wI = FloatVector.fromArray(SPECIES_FLOAT, twImag, j);

            // W^k * (i*B) complex multiply
            FloatVector wibR = wR.mul(ibR).sub(wI.mul(ibI));
            FloatVector wibI = wR.mul(ibI).add(wI.mul(ibR));

            // X[k] = 0.5 * (A - W^k * i*B)
            aR.sub(wibR).mul(half).intoArray(outReal, j);
            aI.sub(wibI).mul(half).intoArray(outImag, j);
        }

        // Scalar tail for unpacking
        for (; j < halfN; j++) {
            int conj = halfN - j;

            float zkR = zReal[j];
            float zkI = zImag[j];
            float zcR = zReal[conj];
            float zcI = -zImag[conj];

            float aR = zkR + zcR;
            float aI = zkI + zcI;

            float bR = zkR - zcR;
            float bI = zkI - zcI;

            float ibR = -bI;
            float ibI = bR;

            float wR = twReal[j];
            float wI = twImag[j];
            float wibR = wR * ibR - wI * ibI;
            float wibI = wR * ibI + wI * ibR;

            outReal[j] = 0.5f * (aR - wibR);
            outImag[j] = 0.5f * (aI - wibI);
        }
    }

    /**
     * Optimized FFT for real-valued input signals (double).
     */
    public static void fftReal(double[] input, double[] outReal, double[] outImag) {
        int n = input.length;
        if ((n & (n - 1)) != 0) throw new IllegalArgumentException("Length must be power of 2");

        int halfN = n / 2;

        // Step 1: Pack N real values into N/2 complex values
        double[] zReal = new double[halfN];
        double[] zImag = new double[halfN];

        // Scalar packing (deinterleaving is complex with current Vector API)
        for (int j = 0; j < halfN; j++) {
            zReal[j] = input[2 * j];
            zImag[j] = input[2 * j + 1];
        }

        // Step 2: Perform N/2 complex FFT (uses Vector API internally)
        fft(zReal, zImag);

        // Step 3: Unpack to get N-point real FFT
        outReal[0] = zReal[0] + zImag[0];
        outImag[0] = 0;

        outReal[halfN] = zReal[0] - zImag[0];
        outImag[halfN] = 0;

        // Precompute twiddle factors and reversed conjugate arrays for vectorization
        double[] twReal = new double[halfN];
        double[] twImag = new double[halfN];
        double[] zRealRev = new double[halfN];
        double[] zImagRev = new double[halfN];
        for (int k = 0; k < halfN; k++) {
            double angle = -Math.PI * k / halfN;
            twReal[k] = Math.cos(angle);
            twImag[k] = Math.sin(angle);
            zRealRev[k] = zReal[halfN - 1 - k];
            zImagRev[k] = -zImag[halfN - 1 - k];
        }

        // Vectorized unpacking (k = 1 to N/2-1)
        int j = 1;
        int loopBound = SPECIES.loopBound(halfN - 1);
        DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);

        for (; j < loopBound + 1; j += SPECIES.length()) {
            if (j + SPECIES.length() > halfN) break;

            // Load Z[k]
            DoubleVector zkR = DoubleVector.fromArray(SPECIES, zReal, j);
            DoubleVector zkI = DoubleVector.fromArray(SPECIES, zImag, j);

            // Load Z*[N/2-k] from pre-reversed arrays
            DoubleVector zcR = DoubleVector.fromArray(SPECIES, zRealRev, j - 1);
            DoubleVector zcI = DoubleVector.fromArray(SPECIES, zImagRev, j - 1);

            // A = Z[k] + Z*[N/2-k]
            DoubleVector aR = zkR.add(zcR);
            DoubleVector aI = zkI.add(zcI);

            // B = Z[k] - Z*[N/2-k]
            DoubleVector bR = zkR.sub(zcR);
            DoubleVector bI = zkI.sub(zcI);

            // i*B = (-bI, bR)
            DoubleVector ibR = bI.neg();
            DoubleVector ibI = bR;

            // Load twiddle factors W^k
            DoubleVector wR = DoubleVector.fromArray(SPECIES, twReal, j);
            DoubleVector wI = DoubleVector.fromArray(SPECIES, twImag, j);

            // W^k * (i*B) complex multiply
            DoubleVector wibR = wR.mul(ibR).sub(wI.mul(ibI));
            DoubleVector wibI = wR.mul(ibI).add(wI.mul(ibR));

            // X[k] = 0.5 * (A - W^k * i*B)
            aR.sub(wibR).mul(half).intoArray(outReal, j);
            aI.sub(wibI).mul(half).intoArray(outImag, j);
        }

        // Scalar tail for unpacking
        for (; j < halfN; j++) {
            int conj = halfN - j;

            double zkR = zReal[j];
            double zkI = zImag[j];
            double zcR = zReal[conj];
            double zcI = -zImag[conj];

            double aR = zkR + zcR;
            double aI = zkI + zcI;

            double bR = zkR - zcR;
            double bI = zkI - zcI;

            double ibR = -bI;
            double ibI = bR;

            double wR = twReal[j];
            double wI = twImag[j];
            double wibR = wR * ibR - wI * ibI;
            double wibI = wR * ibI + wI * ibR;

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