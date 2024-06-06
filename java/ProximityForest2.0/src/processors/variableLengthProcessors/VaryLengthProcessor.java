package processors.variableLengthProcessors;

import application.Application;

import java.util.Random;

public abstract class VaryLengthProcessor {
    public enum PROCESSOR {
        SuffixNoise, PrefixSuffixNoise, PrefixSuffixZero, UniformScaling, NoProcess, SeriesChopper
    }

    public PROCESSOR processorType = PROCESSOR.NoProcess;

    Random random = Application.rand;

    public abstract double[] process(final double[] data, final int maxLen);

    public void setRandomSeed(final int seed) {
        random = new Random(seed);
    }

    public static VaryLengthProcessor getProcessor(PROCESSOR[] processors, int r) {
        return getProcessor(processors[r]);
    }

    public static VaryLengthProcessor getProcessor(PROCESSOR processor) {
        switch (processor) {
            case NoProcess:
                return new NoProcessing();
            case PrefixSuffixNoise:
                return new PrefixSuffixNoisePadder();
            case PrefixSuffixZero:
                return new PrefixSuffixZeroPadder();
            case UniformScaling:
                return new MaxLengthRescaler();
            default:
                return new SuffixNoisePadder();
        }
    }
}
