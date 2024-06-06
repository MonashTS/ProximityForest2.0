package hydra;

import datasets.Sequences;
import transforms.DifferentiationFilter;
import utils.Tools;

import java.util.Arrays;
import java.util.Random;

public class Hydra {
    private double[][][][][][] filters;
    private int[] dilations;
    private int[] paddings;

    private final int kernelLength = 9;
    public int numKernelsPerGroup = 8;
    public int numGroups = 64;
    public int divisor = 2;
    public int h;
    public int numDilations;

    private Random rand;

    private SparseScaler scaler;

    private static class SparseScaler {
        private final int exponent = 4;
        private boolean mask = true;
        private boolean fitted = false;
        private double[] epsilon;
        private double[] mu;
        private double[] sigma;

        private double[][] epsilon2;
        private double[][] mu2;
        private double[][] sigma2;

        public void fit(final double[][] X) {
            final int n = X.length;
            final int l = X[0].length;
            final double[][] out = new double[n][l];

            mu = new double[l];
            sigma = new double[l];
            epsilon = new double[l];
            for (int j = 0; j < l; j++) {
                // calculate mean
                for (int i = 0; i < n; i++) {
                    out[i][j] = Math.sqrt(Math.max(X[i][j], 0));
                    if (out[i][j] == 0) epsilon[j]++;
                    mu[j] += out[i][j];
                }
                mu[j] /= n;

                // calculate std
                for (int i = 0; i < n; i++) {
                    final double diff = out[i][j] - mu[j];
                    sigma[j] += diff * diff;
                }

                // adjust std
                epsilon[j] = Math.pow(epsilon[j] / n, this.exponent) + 1e-8;
                sigma[j] = Math.sqrt(sigma[j] / n) + epsilon[j];
            }
            this.fitted = true;
        }

        public void fit3D(final double[][][] X) {
            final int l1 = X.length;
            final int n = X[0].length;
            final int l2 = X[0][1].length;
            final double[][][] out = new double[l1][n][l2];

            mu2 = new double[l1][l2];
            sigma2 = new double[l1][l2];
            epsilon2 = new double[l1][l2];
            for (int j1 = 0; j1 < l1; j1++) {
                // calculate mean
                for (int j2 = 0; j2 < l2; j2++) {
                    for (int i = 0; i < n; i++) {
                        out[j1][i][j2] = Math.sqrt(Math.max(X[j1][i][j2], 0));
                        if (out[j1][i][j2] == 0) epsilon2[j1][j2]++;
                        mu2[j1][j2] += out[j1][i][j2];
                    }
                    mu2[j1][j2] /= n;

                    // calculate std
                    for (int i = 0; i < n; i++) {
                        final double diff = out[j1][i][j2] - mu2[j1][j2];
                        sigma2[j1][j2] += diff * diff;
                    }

                    // adjust std
                    epsilon2[j1][j2] = Math.pow(epsilon2[j1][j2] / n, this.exponent) + 1e-8;
                    sigma2[j1][j2] = Math.sqrt(sigma2[j1][j2] / n) + epsilon2[j1][j2];
                }
            }
            this.fitted = true;
        }


        public double[][] transform(final double[][] X) {
            final int n = X.length;
            final int l = X[0].length;
            final double[][] out = new double[n][l];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < l; j++) {
                    if (this.mask) {
                        out[i][j] = X[i][j] != 0 ? 1 : 0;
                        out[i][j] = (Math.max(X[i][j], 0) - mu[j]) * out[i][j] / sigma[j];
                    } else {
                        out[i][j] = (X[i][j] - mu[j]) / sigma[j];
                    }
                }
            }
            return out;
        }

        public double[][][] transform3D(final double[][][] X) {
            final int n = X[0].length;
            final int l1 = X.length;
            final int l2 = X[0][1].length;
            final double[][][] out = new double[l1][n][l2];
            for (int i = 0; i < n; i++) {
                for (int j1 = 0; j1 < l1; j1++) {
                    for (int j2 = 0; j2 < l2; j2++) {
                        if (this.mask) {
                            out[j1][i][j2] = X[j1][i][j2] != 0 ? 1 : 0;
                            out[j1][i][j2] = (Math.max(X[j1][i][j2], 0) - mu2[j1][j2]) * out[j1][i][j2] / sigma2[j1][j2];
                        } else {
                            out[j1][i][j2] = (X[j1][i][j2] - mu2[j1][j2]) / sigma2[j1][j2];
                        }
                    }
                }
            }
            return out;
        }

        public double[] transform(final double[] X) {
            final int l = X.length;
            final double[] out = new double[l];
            for (int j = 0; j < l; j++) {
                if (this.mask) {
                    out[j] = X[j] != 0 ? 1 : 0;
                    out[j] = (Math.max(X[j], 0) - mu[j]) * out[j] / sigma[j];
                } else {
                    out[j] = (X[j] - mu[j]) / sigma[j];
                }
            }
            return out;
        }

        public double[][] transform3D(final double[][] X) {
            final int l1 = X.length;
            final int l2 = X[0].length;
            final double[][] out = new double[l1][l2];
            for (int j1 = 0; j1 < l1; j1++) {
                for (int j2 = 0; j2 < l2; j2++) {
                    if (this.mask) {
                        out[j1][j2] = X[j1][j2] != 0 ? 1 : 0;
                        out[j1][j2] = (Math.max(X[j1][j2], 0) - mu2[j1][j2]) * out[j1][j2] / sigma2[j1][j2];
                    } else {
                        out[j1][j2] = (X[j1][j2] - mu2[j1][j2]) / sigma2[j1][j2];
                    }
                }
            }
            return out;
        }

        public double[][] fitTransform(final double[][] X) {
            this.fit(X);
            return transform(X);
        }


        public double[][][] fitTransform3D(final double[][][] X) {
            this.fit3D(X);
            return transform3D(X);
        }
    }

    public Hydra() {
        rand = new Random();
    }


    public void fit(Sequences xTrain) {
        this.generateKernels(xTrain.length(), xTrain.dim());
    }

    public void generateKernels(final int inputLength, final int inputDim) {
        // max_exponent = np.log2((input_length - 1) / (9 - 1))  # kernel length = 9
        final int maxExponent = (int) Tools.log2((double) (inputLength - 1) / (kernelLength - 1));

        // self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        // self.num_dilations = len(self.dilations)
        // self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode="floor").int()
        numDilations = maxExponent + 1;
        paddings = new int[numDilations];
        dilations = Tools.arange(numDilations);
        for (int i = 0; i < numDilations; i++) {
            dilations[i] = (int) Math.pow(2, dilations[i]);
            paddings[i] = (kernelLength - 1) * dilations[i] / 2;
        }

        // self.divisor = min(2, self.g)
        this.divisor = Math.min(2, this.numGroups);
        // self.h = self.g // self.divisor
        this.h = this.numGroups / this.divisor;

        // self.W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        filters = new double[numDilations][divisor][h][numKernelsPerGroup][inputDim][kernelLength];
        for (int i = 0; i < numDilations; i++) {
            for (int j = 0; j < divisor; j++) {
                for (int h = 0; h < this.h; h++) {
                    for (int k = 0; k < numKernelsPerGroup; k++) {
                        for (int l = 0; l < inputDim; l++) {
                            // calculate the mean of each filter
                            double meanWeights = 0;
                            for (int m = 0; m < kernelLength; m++) {
                                filters[i][j][h][k][l][m] = rand.nextGaussian();
                                meanWeights += filters[i][j][h][k][l][m];
                            }
                            meanWeights /= kernelLength;

                            // calculate the absolute value of each filter
                            double absSum = 0;
                            for (int m = 0; m < kernelLength; m++) {
                                // self.W = self.W - self.W.mean(-1, keepdims=True)
                                filters[i][j][h][k][l][m] = filters[i][j][h][k][l][m] - meanWeights;
                                absSum += Math.abs(filters[i][j][h][k][l][m]);
                            }
                            for (int m = 0; m < kernelLength; m++) {
                                // self.W = self.W / self.W.abs().sum(-1, keepdims=True)
                                filters[i][j][h][k][l][m] /= absSum;
                            }
                        }
                    }
                }
            }
        }
    }

    public double[][] transform(final Sequences inputs) {
        final int numExamples = inputs.size();

        final Sequences diffInputs = DifferentiationFilter.getDiff(inputs, 1);

        double[][] outputs = new double[numExamples][numDilations * divisor * numKernelsPerGroup * h * 2];
        for (int i = 0; i < inputs.size(); i++) {
            int featureCounter = 0;
            for (int dilationIndex = 0; dilationIndex < this.numDilations; dilationIndex++) {
                final int d = this.dilations[dilationIndex];
                final int p = this.paddings[dilationIndex];

                // for each representation, raw or diff
                for (int diffIndex = 0; diffIndex < this.divisor; diffIndex++) {
                    double[][] series;
                    if (diffIndex == 0) series = inputs.get(i).get();
                    else series = diffInputs.get(i).get();

                    if (p > 0) {
                        final int inputLength = series[0].length;
                        final double[][] newSeries = new double[series.length][inputLength + p + p];
                        for (int j = 0; j < series.length; j++)
                            System.arraycopy(series[j], 0, newSeries[j], p, inputLength);
                        series = newSeries;
                    }

                    final int inputLength = series[0].length;
                    final int outputLength = inputLength - ((kernelLength - 1) * d);

                    for (int groupIndex = 0; groupIndex < this.h; groupIndex++) {
                        final double[] maxs = new double[outputLength];
                        final double[] mins = new double[outputLength];

                        final int[] maxIndex = new int[outputLength];
                        final int[] minIndex = new int[outputLength];

                        Arrays.fill(maxs, Double.NEGATIVE_INFINITY);
                        Arrays.fill(mins, Double.POSITIVE_INFINITY);

                        for (int kernelIndex = 0; kernelIndex < this.numKernelsPerGroup; kernelIndex++) {
                            for (int j = 0; j < outputLength; j++) {
                                double sum = 0;
                                for (int k = 0; k < series.length; k++) {
                                    final double[] filter = this.filters[dilationIndex][diffIndex][groupIndex][kernelIndex][k];
                                    for (int l = 0; l < filter.length; l++)
                                        sum += filter[l] * series[k][j + (l * d)];
                                }
                                if (sum > maxs[j]) {
                                    maxs[j] = sum;
                                    maxIndex[j] = kernelIndex;
                                }
                                if (sum < mins[j]) {
                                    mins[j] = sum;
                                    minIndex[j] = kernelIndex;
                                }
                            }
                        }
                        final int[] maxCounts = new int[numKernelsPerGroup];
                        final int[] minCounts = new int[numKernelsPerGroup];
                        for (int j = 0; j < outputLength; j++) {
                            maxCounts[maxIndex[j]]++;
                            minCounts[minIndex[j]]++;
                        }
                        for (int j = 0; j < numKernelsPerGroup; j++) {
                            outputs[i][featureCounter] = maxCounts[j];
                            outputs[i][featureCounter + 1] = minCounts[j];
                            featureCounter += 2;
                        }
                    }
                }
            }
        }

        if (scaler == null) {
            scaler = new SparseScaler();
            outputs = scaler.fitTransform(outputs);
        } else {
            outputs = scaler.transform(outputs);
        }
        return outputs;
    }

    public double[] transform(final double[] input) {
        final double[] diffInput = DifferentiationFilter.getDiff(input, 1);

        double[] outputs = new double[numDilations * divisor * numKernelsPerGroup * h * 2];
        int featureCounter = 0;
        for (int dilationIndex = 0; dilationIndex < this.numDilations; dilationIndex++) {
            final int d = this.dilations[dilationIndex];
            final int p = this.paddings[dilationIndex];

            for (int diffIndex = 0; diffIndex < this.divisor; diffIndex++) {
                double[] series;
                if (diffIndex == 0) series = input;
                else series = diffInput;

                if (p > 0) {
                    final int inputLength = series.length;
                    final double[] newSeries = new double[inputLength + p + p];
                    System.arraycopy(series, 0, newSeries, p, inputLength);
                    series = newSeries;
                }

                final int inputLength = series.length;
                final int outputLength = inputLength - ((kernelLength - 1) * d);

                for (int groupIndex = 0; groupIndex < this.h; groupIndex++) {
                    final double[] maxs = new double[outputLength];
                    final double[] mins = new double[outputLength];

                    final int[] maxIndex = new int[outputLength];
                    final int[] minIndex = new int[outputLength];

                    Arrays.fill(maxs, Double.NEGATIVE_INFINITY);
                    Arrays.fill(mins, Double.POSITIVE_INFINITY);

                    for (int kernelIndex = 0; kernelIndex < this.numKernelsPerGroup; kernelIndex++) {
                        for (int j = 0; j < outputLength; j++) {
                            double sum = 0;
                            final double[] filter = this.filters[dilationIndex][diffIndex][groupIndex][kernelIndex][0];
                            for (int l = 0; l < filter.length; l++)
                                sum += filter[l] * series[j + (l * d)];

                            if (sum > maxs[j]) {
                                maxs[j] = sum;
                                maxIndex[j] = kernelIndex;
                            }
                            if (sum < mins[j]) {
                                mins[j] = sum;
                                minIndex[j] = kernelIndex;
                            }
                        }
                    }
                    final int[] maxCounts = new int[numKernelsPerGroup];
                    final int[] minCounts = new int[numKernelsPerGroup];
                    for (int j = 0; j < outputLength; j++) {
                        maxCounts[maxIndex[j]]++;
                        minCounts[minIndex[j]]++;
                    }
                    for (int j = 0; j < numKernelsPerGroup; j++) {
                        outputs[featureCounter] = maxCounts[j];
                        outputs[featureCounter + 1] = minCounts[j];
                        featureCounter += 2;
                    }
                }
            }
        }

        outputs = scaler.transform(outputs);

        return outputs;
    }

    public double[][][] transformByGroup(final Sequences inputs) {
        final int numExamples = inputs.size();

        final Sequences diffInputs = DifferentiationFilter.getDiff(inputs, 1);

        double[][][] outputs = new double[h][numExamples][numDilations * divisor * numKernelsPerGroup * 2];

        for (int i = 0; i < inputs.size(); i++) {
            int[] featureCounts = new int[h];
            for (int dilationIndex = 0; dilationIndex < this.numDilations; dilationIndex++) {
                final int d = this.dilations[dilationIndex];
                final int p = this.paddings[dilationIndex];

                // for each representation, raw or diff
                for (int diffIndex = 0; diffIndex < this.divisor; diffIndex++) {
                    double[][] series;
                    if (diffIndex == 0) series = inputs.get(i).get();
                    else series = diffInputs.get(i).get();

                    if (p > 0) {
                        final int inputLength = series[0].length;
                        final double[][] newSeries = new double[series.length][inputLength + p + p];
                        for (int j = 0; j < series.length; j++)
                            System.arraycopy(series[j], 0, newSeries[j], p, inputLength);
                        series = newSeries;
                    }

                    final int inputLength = series[0].length;
                    final int outputLength = inputLength - ((kernelLength - 1) * d);

                    for (int groupIndex = 0; groupIndex < this.h; groupIndex++) {
                        final double[] maxs = new double[outputLength];
                        final double[] mins = new double[outputLength];

                        final int[] maxIndex = new int[outputLength];
                        final int[] minIndex = new int[outputLength];

                        Arrays.fill(maxs, Double.NEGATIVE_INFINITY);
                        Arrays.fill(mins, Double.POSITIVE_INFINITY);

                        for (int kernelIndex = 0; kernelIndex < this.numKernelsPerGroup; kernelIndex++) {
                            for (int j = 0; j < outputLength; j++) {
                                double sum = 0;
                                for (int k = 0; k < series.length; k++) {
                                    final double[] filter = this.filters[dilationIndex][diffIndex][groupIndex][kernelIndex][k];
                                    for (int l = 0; l < filter.length; l++)
                                        sum += filter[l] * series[k][j + (l * d)];
                                }
                                if (sum > maxs[j]) {
                                    maxs[j] = sum;
                                    maxIndex[j] = kernelIndex;
                                }
                                if (sum < mins[j]) {
                                    mins[j] = sum;
                                    minIndex[j] = kernelIndex;
                                }
                            }
                        }
                        final int[] maxCounts = new int[numKernelsPerGroup];
                        final int[] minCounts = new int[numKernelsPerGroup];
                        for (int j = 0; j < outputLength; j++) {
                            maxCounts[maxIndex[j]]++;
                            minCounts[minIndex[j]]++;
                        }
                        for (int j = 0; j < numKernelsPerGroup; j++) {
                            outputs[groupIndex][i][featureCounts[groupIndex]] = maxCounts[j];
                            outputs[groupIndex][i][featureCounts[groupIndex] + 1] = minCounts[j];
                            featureCounts[groupIndex] += 2;
                        }
                    }
                }
            }
            // System.out.println(featureCounter);
        }
        if (scaler == null) {
            scaler = new SparseScaler();
            outputs = scaler.fitTransform3D(outputs);
        } else {
            outputs = scaler.transform3D(outputs);
        }
        return outputs;
    }

    public double[][] transformByGroup(final double[] input) {
        final double[] diffInput = DifferentiationFilter.getDiff(input, 1);

        double[][] outputs = new double[h][numDilations * divisor * numKernelsPerGroup * 2];
        int[] featureCounts = new int[h];
        for (int dilationIndex = 0; dilationIndex < this.numDilations; dilationIndex++) {
            final int d = this.dilations[dilationIndex];
            final int p = this.paddings[dilationIndex];

            for (int diffIndex = 0; diffIndex < this.divisor; diffIndex++) {
                double[] series;
                if (diffIndex == 0) series = input;
                else series = diffInput;

                if (p > 0) {
                    final int inputLength = series.length;
                    final double[] newSeries = new double[inputLength + p + p];
                    System.arraycopy(series, 0, newSeries, p, inputLength);
                    series = newSeries;
                }

                final int inputLength = series.length;
                final int outputLength = inputLength - ((kernelLength - 1) * d);

                for (int groupIndex = 0; groupIndex < this.h; groupIndex++) {
                    final double[] maxs = new double[outputLength];
                    final double[] mins = new double[outputLength];

                    final int[] maxIndex = new int[outputLength];
                    final int[] minIndex = new int[outputLength];

                    Arrays.fill(maxs, Double.NEGATIVE_INFINITY);
                    Arrays.fill(mins, Double.POSITIVE_INFINITY);

                    for (int kernelIndex = 0; kernelIndex < this.numKernelsPerGroup; kernelIndex++) {
                        for (int j = 0; j < outputLength; j++) {
                            double sum = 0;
                            final double[] filter = this.filters[dilationIndex][diffIndex][groupIndex][kernelIndex][0];
                            for (int l = 0; l < filter.length; l++)
                                sum += filter[l] * series[j + (l * d)];

                            if (sum > maxs[j]) {
                                maxs[j] = sum;
                                maxIndex[j] = kernelIndex;
                            }
                            if (sum < mins[j]) {
                                mins[j] = sum;
                                minIndex[j] = kernelIndex;
                            }
                        }
                    }
                    final int[] maxCounts = new int[numKernelsPerGroup];
                    final int[] minCounts = new int[numKernelsPerGroup];
                    for (int j = 0; j < outputLength; j++) {
                        maxCounts[maxIndex[j]]++;
                        minCounts[minIndex[j]]++;
                    }
                    for (int j = 0; j < numKernelsPerGroup; j++) {
                        outputs[groupIndex][featureCounts[groupIndex]] = maxCounts[j];
                        outputs[groupIndex][featureCounts[groupIndex] + 1] = minCounts[j];
                        featureCounts[groupIndex] += 2;
                    }
                }
            }
        }

        outputs = scaler.transform3D(outputs);

        return outputs;
    }
}
