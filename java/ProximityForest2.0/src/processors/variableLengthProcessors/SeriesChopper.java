package processors.variableLengthProcessors;

public class SeriesChopper extends VaryLengthProcessor {
    public SeriesChopper() {
        this.processorType = PROCESSOR.SeriesChopper;
    }

    @Override
    public double[] process(final double[] data, final int minLen) {
        final double[] scaledData = new double[minLen];

        System.arraycopy(data, 0, scaledData, 0, minLen);

        return scaledData;
    }
}
