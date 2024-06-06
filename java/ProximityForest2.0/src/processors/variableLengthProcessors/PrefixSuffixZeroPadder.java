package processors.variableLengthProcessors;

public class PrefixSuffixZeroPadder extends VaryLengthProcessor {
    public PrefixSuffixZeroPadder() {
        processorType = PROCESSOR.PrefixSuffixZero;
    }

    @Override
    public double[] process(final double[] data, final int maxLen) {
        final int seqLen = data.length;
        final double[] arr = new double[seqLen + 2];
        final int lastIndex = seqLen + 1;
        arr[0] = 0;
        arr[lastIndex] = 0;
        System.arraycopy(data, 0, arr, 1, lastIndex - 1);
        return arr;
    }
}
