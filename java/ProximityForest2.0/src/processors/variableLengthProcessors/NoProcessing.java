package processors.variableLengthProcessors;

public class NoProcessing extends VaryLengthProcessor {
    public NoProcessing() {
        processorType = PROCESSOR.NoProcess;
    }

    @Override
    public double[] process(final double[] data, final int maxLen) {
        return data;
    }
}
