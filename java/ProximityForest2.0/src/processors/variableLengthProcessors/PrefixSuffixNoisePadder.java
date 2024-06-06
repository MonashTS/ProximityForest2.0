package processors.variableLengthProcessors;

public class PrefixSuffixNoisePadder extends VaryLengthProcessor {
    public PrefixSuffixNoisePadder() {
        processorType = PROCESSOR.PrefixSuffixNoise;
    }

    @Override
    public double[] process(final double[] data, final int maxLen) {
        final int seqLen = data.length;
        final int diffLen = (int) (0.5 * (maxLen - seqLen));
        final double[] arr = new double[maxLen];

        for (int i = 0; i < diffLen; i++) {
            final double val = random.nextDouble() / 1000;
            arr[i] = val;
        }

        System.arraycopy(data, 0, arr, diffLen, seqLen);

        for (int i = seqLen + diffLen; i < maxLen; i++) {
            final double val = random.nextDouble() / 1000;
            arr[i] = val;
        }
        return arr;
    }
}
