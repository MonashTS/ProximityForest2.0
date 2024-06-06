package processors.variableLengthProcessors;

import utils.Tools;

public class SuffixNoisePadder extends VaryLengthProcessor {
    public SuffixNoisePadder() {
        this.processorType = PROCESSOR.SuffixNoise;
    }

    @Override
    public double[] process(final double[] data, final int maxLen) {
        final int seqLen = data.length;
        final double[] arr = new double[maxLen];

        System.arraycopy(data, 0, arr, 0, seqLen);
        // iterate from the back
        int i = maxLen - 1;
        while (Tools.isMissing(arr[i])) {
            arr[i] = random.nextDouble() / 1000;
            i--;
        }
        return arr;
    }
}
