package transforms;

import datasets.Sequence;
import datasets.Sequences;

import static transforms.Transforms.TimeSeriesTransforms.d1;

public class DerivativeFilter {
    public static Sequences getFirstDerivative(final Sequences data) {
        final Sequences output = new Sequences(data.size());

        for (int i = 0; i < data.size(); i++) {
            final Sequence s = new Sequence(getFirstDerivative(data.get(i).data[0]), data.get(i).classLabel);
            s.type = d1;
            output.add(s, i);
        }
        return output;
    }

    public static double[] getFirstDerivative(final double[] input) {
        final double[] derivative = new double[input.length];

        for (int i = 1; i < input.length - 1; i++) {
            derivative[i] = ((input[i] - input[i - 1]) + ((input[i + 1] - input[i - 1]) / 2)) / 2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length - 1] = derivative[derivative.length - 2];

        return derivative;
    }

    public static double[][] getDerivative(final double[][] data, final int derivative) {
        final double[][] output = data.clone();
        for (int d = 0; d < derivative; d++) {
            for (int i = 0; i < data.length; i++) {
                output[i] = getFirstDerivative(data[i]);
            }
        }
        return output;
    }

    public static Sequences getDerivative(final Sequences data, final int derivative) {
        final Sequences output = new Sequences(data.size());

        for (int i = 0; i < data.size(); i++) {
            double[] s = data.get(i).firstChannel();
            for (int d = 0; d < derivative; d++)
                s = getFirstDerivative(s);
            // todo add type to sequence
            output.add(new Sequence(s, data.get(i).classLabel), i);
        }
        return output;
    }
}
