package org.example;

import org.jtransforms.fft.FloatFFT_1D;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import static java.lang.Math.pow;

/**
 * Copied from jtransforms. Adjusted for VectorFFT
 *
 * This is a series of JUnit tests for the {@link FloatFFT_1D}. First,
 * {@link FloatFFT_1D#complexForward(float[])} is tested by comparison with
 * reference data (FFTW). Then the other methods of this class are tested using
 * {@link FloatFFT_1D#complexForward(float[])} as a reference.
 *
 * @author S&eacute;bastien Brisard
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class FloatFFT_1DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "FFT of size %d: ";

    /**
     * Name of binary files (input, untransformed data).
     */
    private final static String FFTW_INPUT_PATTERN = "fftw%d.in";

    /**
     * Name of binary files (output, transformed data).
     */
    private final static String FFTW_OUTPUT_PATTERN = "fftw%d.out";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110602;

    private static final double EPS = pow(10, -4);

    @Parameters
    public static Collection<Object[]> getParameters()
    {
        // VectorFFT only supports power-of-2 sizes
        final int[] size = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                            8192, 16384, 32768, 65536, 131072};

        final ArrayList<Object[]> parameters = new ArrayList<Object[]>();
        for (int i = 0; i < size.length; i++) {
            parameters.add(new Object[]{size[i], SEED});
        }
        return parameters;
    }

    /**
     * The size of the FFT to be tested.
     */
    private final int n;

    /**
     * For the generation of the data arrays.
     */
    private final Random random;

    /**
     * Creates a new instance of this class.
     *
     * @param n the size of the FFT to be tested
     * @param seed the seed of the random generator
     */
    public FloatFFT_1DTest(final int n, final long seed)
    {
        this.n = n;
        this.random = new Random(seed);
    }

    /**
     * Read the binary reference data files generated with FFTW. The structure
     * of these files is very simple: double values are written linearly (little
     * endian).
     *
     * @param name the file name
     * @param data the array to be updated with the data read (the size of this
     *             array gives the number of <code>double</code> to be retrieved
     */
    public void readData(final String name, final double[] data)
    {
        try {
            final File f = new File(getClass().getClassLoader()
                .getResource(name).getFile());
            final FileInputStream fin = new FileInputStream(f);
            final FileChannel fc = fin.getChannel();
            final ByteBuffer buffer = ByteBuffer.allocate(8 * data.length);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            fc.read(buffer);
            for (int i = 0; i < data.length; i++) {
                data[i] = buffer.getDouble(8 * i);
            }
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    /**
     * This is a test of {@link VectorFFT#fft(float[], float[])}. This
     * method is tested by computation of the FFT of some pre-generated data,
     * and comparison with results obtained with FFTW.
     */
    @Test
    public void testComplexForward()
    {
        // Read interleaved input data from FFTW (stored as doubles)
        final double[] interleavedInput = new double[2 * n];
        readData(String.format(FFTW_INPUT_PATTERN, n), interleavedInput);

        // Convert interleaved doubles to split float real/imag arrays for VectorFFT
        final float[] real = new float[n];
        final float[] imag = new float[n];
        for (int i = 0; i < n; i++) {
            real[i] = (float) interleavedInput[2 * i];
            imag[i] = (float) interleavedInput[2 * i + 1];
        }

        // Read expected output from FFTW (stored as doubles)
        final double[] expectedDouble = new double[2 * n];
        readData(String.format(FFTW_OUTPUT_PATTERN, n), expectedDouble);

        // Convert to float for comparison
        final float[] expected = new float[2 * n];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = (float) expectedDouble[i];
        }

        // Run VectorFFT
        VectorFFT.fft(real, imag);

        // Convert result back to interleaved for comparison
        final float[] actual = new float[2 * n];
        for (int i = 0; i < n; i++) {
            actual[2 * i] = real[i];
            actual[2 * i + 1] = imag[i];
        }

        // Compute RMSE
        double rmse = computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link VectorFFT#fftReal(float[], float[], float[])}.
     * Tests real-valued FFT by comparing with complex FFT using zero imaginary input.
     */
    @Test
    public void testRealForward()
    {
        // Generate random real input
        final float[] realInput = new float[n];
        for (int i = 0; i < n; i++) {
            realInput[i] = 2.f * random.nextFloat() - 1.f;
        }

        // Compute reference using complex FFT with zero imaginary
        final float[] refReal = new float[n];
        final float[] refImag = new float[n];
        System.arraycopy(realInput, 0, refReal, 0, n);
        // refImag is already zeros
        VectorFFT.fft(refReal, refImag);

        // Compute using optimized real FFT
        final float[] outReal = new float[n / 2 + 1];
        final float[] outImag = new float[n / 2 + 1];
        VectorFFT.fftReal(realInput, outReal, outImag);

        // Compare first N/2+1 values (the unique part of the spectrum)
        double sumSq = 0.0;
        for (int i = 0; i <= n / 2; i++) {
            double diffR = outReal[i] - refReal[i];
            double diffI = outImag[i] - refImag[i];
            sumSq += diffR * diffR + diffI * diffI;
        }
        double rmse = Math.sqrt(sumSq / (n / 2 + 1));

        Assert.assertEquals(String.format(DEFAULT_MESSAGE, n) + "realForward, rmse = " + rmse, 0.0, rmse, EPS);
    }

    private static double computeRMSE(float[] actual, float[] expected) {
        double sumSq = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double diff = actual[i] - expected[i];
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / actual.length);
    }
}
