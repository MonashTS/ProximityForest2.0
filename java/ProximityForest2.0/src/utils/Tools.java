package utils;

import datasets.Sequence;
import datasets.Sequences;

public class Tools {
    public static double EPSILON = 10e-12;

    public static double cost(double a, double b) {
        double d = a - b;
        return d * d;
    }

    public static double absCost(double a, double b) {
        double d = a - b;
        return Math.abs(d);
    }

    public static double sqrtCost(double a, double b) {
        double d = Math.abs(a - b);
        return Math.sqrt(d);
    }

    public static double cost(double a, double b, double ge) {
        double d = a - b;
        if (ge == 1) return absCost(a, b);
        if (ge == 0.5) return sqrtCost(a, b);
        if (ge == 2) return cost(a, b);

        return Math.pow(Math.abs(d), ge);
    }

    public static int argMin3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
    }

    public static int min(int a, int b, int c) {
        return Integer.min(a, Integer.min(b, c));
    }

    public static int max(int a, int b, int c) {
        return Integer.max(a, Integer.max(b, c));
    }

    public static double min(double a, double b, double c) {
        return Double.min(a, Double.min(b, c));
    }

    public static double max(double a, double b, double c) {
        return Double.max(a, Double.max(b, c));
    }


    public static String doTime(double elapsedTimeNanoSeconds) {
        final double duration = elapsedTimeNanoSeconds / 1e6;
        return String.format("%d s %.3f ms", (int) (duration / 1000), (duration % 1000));
    }

    public static String doTimeNs(double elapsedTimeNanoSeconds) {
        int hour = (int) (elapsedTimeNanoSeconds / 3.6e+12);
        int min = (int) (elapsedTimeNanoSeconds / 6e+10);
        int s = (int) (elapsedTimeNanoSeconds / 1e9);
        int ms = (int) (elapsedTimeNanoSeconds / 1e6);
        int us = (int) (elapsedTimeNanoSeconds / 1e3);
        StringBuilder str = new StringBuilder();
        if (hour > 0) str.append((hour % 60)).append(" H ");
        if (min > 0) str.append((min % 60)).append(" M ");
        if (s > 0) str.append((s % 60)).append(" s ");
        if (ms > 0) str.append((ms % 1000)).append(" ms ");
        if (us > 0) str.append((us % 1000)).append(" us ");

        str.append(((int) (elapsedTimeNanoSeconds % 1000))).append(" ns");

        return str.toString();
    }

    public static boolean isMissing(double a) {
        return Double.isNaN(a);
    }

    public static double linearInterp(final double t, final double y0, final int t0, final double y1, final int t1) {
        return y0 + (t - t0) * (y1 - y0) / (t1 - t0);
    }

    public static double stdv_p(Sequences input) {
        double sumx = 0;
        double sumx2 = 0;
        for (Sequence ins2array : input.data) {
            for (double v : ins2array.firstChannel()) {//-1 to avoid classVal
                sumx += v;
                sumx2 += v * v;
            }
        }
        int n = input.size() * input.length();
        double mean = sumx / n;
        return Math.sqrt(sumx2 / (n) - mean * mean);
    }

    public static int[] getInclusive10(final int min, final int max) {
        int[] output = new int[10];

        double diff = 1.0 * (max - min) / 9;
        double[] doubleOut = new double[10];
        doubleOut[0] = min;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            doubleOut[i] = doubleOut[i - 1] + diff;
            output[i] = (int) Math.round(doubleOut[i]);
        }
        output[9] = max; // to make sure max isn't omitted due to double imprecision
        return output;
    }

    public static double[] getInclusive10(final double min, final double max) {
        double[] output = new double[10];
        double diff = 1.0 * (max - min) / 9;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            output[i] = output[i - 1] + diff;
        }
        output[9] = max;

        return output;
    }


    public static int[] generateOutputs(int[] classCounts) {
        double bsfCount = -1;
        final int[] out = new int[classCounts.length + 1];
        System.arraycopy(classCounts, 0, out, 1, classCounts.length);
        out[0] = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > bsfCount) {
                bsfCount = classCounts[i];
                out[0] = i;
            }
        }

        return out;
    }

    public static int[] getDatasetIndices(Sequences dataset) {
        int[] indices = new int[dataset.size()];
        for (int i = 0; i < indices.length; i++)
            indices[i] = i;
        return indices;
    }

    public static int[] arange(final int size) {
        int[] indices = new int[size];
        for (int i = 0; i < indices.length; i++)
            indices[i] = i;
        return indices;
    }

    public static double log2(double N) {
        // calculate log2 N indirectly
        // using log() method

        return Math.log(N) / Math.log(2);
    }

    public static double[] doubleLinspace(final double start, final double stop, final int num) {
        final double[] out = new double[num];
        final double step = (stop - start) / (num - 1);
        double val = start;
        for (int j = 0; j < num; j++) {
            out[j] = Math.min(val, stop);
            val += step;
        }

        return out;
    }

}
