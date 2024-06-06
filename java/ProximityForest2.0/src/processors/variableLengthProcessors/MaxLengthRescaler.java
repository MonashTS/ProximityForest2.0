package processors.variableLengthProcessors;

public class MaxLengthRescaler extends VaryLengthProcessor {
    public MaxLengthRescaler() {
        this.processorType = PROCESSOR.UniformScaling;
    }

    @Override
    public double[] process(final double[] data, final int maxLen) {
        final int seqLen = data.length;
        final double[] scaledData = new double[maxLen];

        for (int j = 0; j < maxLen; j++) {
            final int scalingFactor = (int) (1.0 * j * seqLen / maxLen);
            scaledData[j] = data[scalingFactor];
        }

        return scaledData;
    }
}
