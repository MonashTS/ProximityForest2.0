package transforms;

import datasets.Sequence;
import datasets.Sequences;

public class DifferentiationFilter {
    public static double[] getDiff(final double[] input) {
        final double[] output = new double[input.length - 1];
        int i;
        for (i = 1; i < input.length - 1; i++)
            output[i - 1] = input[i] - input[i - 1];
        output[i - 1] = input[i] - input[i - 1];
        return output;
    }

    public static double[][] getDiff(final double[][] data, final int n) {
        final double[][] output = data.clone();
        for (int d = 0; d < n; d++) {
            for (int i = 0; i < data.length; i++) {
                output[i] = getDiff(data[i]);
            }
        }
        return output;
    }

    public static Sequences getDiff(final Sequences data, final int n) {
        final Sequences output = new Sequences(data.size());

        for (int i = 0; i < data.size(); i++) {
            double[] ss = data.get(i).firstChannel();
            for (int d = 0; d < n; d++)
                ss = getDiff(ss);
            final Sequence s = new Sequence(ss, data.get(i).classLabel);
            if (n == 1) s.type = Transforms.TimeSeriesTransforms.diff1;
            else if (n == 2) s.type = Transforms.TimeSeriesTransforms.diff2;
            output.add(s, i);
        }
        return output;
    }

    public static double[] getDiff(final double[] data, final int n) {
        double[] s = data.clone();
        for (int d = 0; d < n; d++)
            s = getDiff(s);

        return s;
    }
}
