package tree.results;

import tree.splitters.NodeSplitter;

import java.util.HashMap;

public class PFResults extends TreeResults {
    public double hydraTransformTime;

    public transient HashMap<NodeSplitter.SplitterType, Integer> splitterCount = new HashMap<>();
    public transient HashMap<NodeSplitter.SplitterType, Long> splitterTime = new HashMap<>();

    public int leafCount = 0;

    public PFResults() {
        super();
    }

    public PFResults(final int[] index, final int nCorrect, final int[] predictions, final int[][] classCounts) {
        super(index, nCorrect, predictions, classCounts);
    }

    public PFResults(final int nCorrect, final double acc, final int[] predictions, final int[][] classCounts) {
        super(nCorrect, acc, predictions, classCounts);
    }

    public PFResults(final int[] index, final int nCorrect, final int[] predictions, final int[][] classCounts, final String trainTest) {
        super(index, nCorrect, predictions, classCounts, trainTest);
    }

    public PFResults(final int nCorrect, final double acc, final int[] predictions, final int[][] classCounts, final String trainTest) {
        super(nCorrect, acc, predictions, classCounts, trainTest);
    }

    public void setHydraTransformTime(final long startTimeNano, final long stopTimeNano) {
        this.hydraTransformTime = (stopTimeNano - startTimeNano);
    }
}
