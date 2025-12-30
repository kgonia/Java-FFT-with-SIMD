package org.example;

public class IterativeFFT {

    public static class Complex {
        double re, im;

        public Complex(double r, double i) { re = r; im = i; }

        public String toString() { return String.format("(%.2f, %.2f)", re, im); }
    }

    public static class ComplexF {
        float re, im;

        public ComplexF(float r, float i) { re = r; im = i; }

        public String toString() { return String.format("(%.2f, %.2f)", re, im); }
    }

    // In-place Iterative FFT
    public static void fft(Complex[] a) {
        int n = a.length;

        // 1. Bit-Reversal Permutation
        // Reorders the array so we can process it bottom-up
        int shift = 1 + Integer.numberOfLeadingZeros(n);
        for (int k = 0; k < n; k++) {
            int j = Integer.reverse(k) >>> shift;
            if (j > k) {
                Complex temp = a[j];
                a[j] = a[k];
                a[k] = temp;
            }
        }

        // 2. The Butterfly Loops (Bottom-Up)
        // len = length of the sub-problem being merged (2, 4, 8... N)
        for (int len = 2; len <= n; len <<= 1) {

            // Angle for the roots of unity for this length
            double angle = -2 * Math.PI / len;

            // Base Twiddle Factor for this length
            // wlen = cos(angle) + i*sin(angle)
            double wlen_re = Math.cos(angle);
            double wlen_im = Math.sin(angle);

            // Process blocks of size 'len'
            for (int i = 0; i < n; i += len) {

                // 'w' starts at 1 (0 radians)
                double w_re = 1;
                double w_im = 0;

                // Butterfly operations within the block
                for (int j = 0; j < len / 2; j++) {

                    // Indices of the pair to combine
                    int u = i + j;
                    int v = i + j + len / 2;

                    // Calculate: w * a[v]
                    double v_re = w_re * a[v].re - w_im * a[v].im;
                    double v_im = w_re * a[v].im + w_im * a[v].re;

                    // Butterfly:
                    // a[u] = u + w*v
                    // a[v] = u - w*v
                    double u_re = a[u].re;
                    double u_im = a[u].im;

                    a[u].re = u_re + v_re;
                    a[u].im = u_im + v_im;

                    a[v].re = u_re - v_re;
                    a[v].im = u_im - v_im;

                    // Rotate 'w' for the next iteration: w = w * wlen
                    double temp_w_re = w_re * wlen_re - w_im * wlen_im;
                    w_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = temp_w_re;
                }
            }
        }
    }

    // In-place Iterative FFT (float version)
    public static void fft(ComplexF[] a) {
        int n = a.length;

        // 1. Bit-Reversal Permutation
        int shift = 1 + Integer.numberOfLeadingZeros(n);
        for (int k = 0; k < n; k++) {
            int j = Integer.reverse(k) >>> shift;
            if (j > k) {
                ComplexF temp = a[j];
                a[j] = a[k];
                a[k] = temp;
            }
        }

        // 2. The Butterfly Loops (Bottom-Up)
        for (int len = 2; len <= n; len <<= 1) {

            float angle = (float) (-2 * Math.PI / len);

            float wlen_re = (float) Math.cos(angle);
            float wlen_im = (float) Math.sin(angle);

            for (int i = 0; i < n; i += len) {

                float w_re = 1;
                float w_im = 0;

                for (int j = 0; j < len / 2; j++) {

                    int u = i + j;
                    int v = i + j + len / 2;

                    float v_re = w_re * a[v].re - w_im * a[v].im;
                    float v_im = w_re * a[v].im + w_im * a[v].re;

                    float u_re = a[u].re;
                    float u_im = a[u].im;

                    a[u].re = u_re + v_re;
                    a[u].im = u_im + v_im;

                    a[v].re = u_re - v_re;
                    a[v].im = u_im - v_im;

                    float temp_w_re = w_re * wlen_re - w_im * wlen_im;
                    w_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = temp_w_re;
                }
            }
        }
    }

    static void main() {
        // Must be power of 2
        int N = 8;

        // Double precision FFT
        Complex[] dataD = new Complex[N];
        for (int i = 0; i < N; i++) dataD[i] = new Complex(i < 4 ? 1 : 0, 0);

        IO.print("Double precision FFT:");
        fft(dataD);
        for (Complex c : dataD) IO.print(c);

        // Float precision FFT
        ComplexF[] dataF = new ComplexF[N];
        for (int i = 0; i < N; i++) dataF[i] = new ComplexF(i < 4 ? 1 : 0, 0);

        IO.print("\nFloat precision FFT:");
        fft(dataF);
        for (ComplexF c : dataF) IO.print(c);
    }
}