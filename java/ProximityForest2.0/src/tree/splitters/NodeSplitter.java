package tree.splitters;

import application.Application;
import datasets.Sequence;
import tree.Node;
import tree.results.NodeSplitterResult;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public abstract class NodeSplitter {
    public String name = "";
    protected Random rand;
    protected transient HashMap<Integer, Integer> exemplars;
    protected final List<Integer> closestNodes = new ArrayList<>();

    public enum SplitterType {
        PF,
        PF2,
    }

    public boolean isTrain = false;

    protected transient Node node;
    protected SplitterType splitterType;
    protected String splitterClass;

    public NodeSplitter() {

    }

    public NodeSplitter(final Node node) {
        this.node = node;
        if (node != null)
            this.rand = Application.rand;
        else
            rand = this.node.rand;
    }

    public abstract NodeSplitterResult fit(final HashMap<Integer, int[]> dataPerClass, final int[] nodeIndices) throws Exception;

    public abstract NodeSplitterResult split(final int[] nodeIndices) throws Exception;

    public abstract int predict(final Sequence query);

    public double gini(final ArrayList<Integer> indices, final int[] labels) {
        double sum = 0.0;
        double p;
        final int totalSize = indices.size();

        final HashMap<Integer, Integer> classDistribution = new HashMap<>();
        for (Integer index : indices) {
            final int label = labels[index];

            if (classDistribution.containsKey(label)) {
                classDistribution.put(label, classDistribution.get(label) + 1);
            } else {
                classDistribution.put(label, 1);
            }
        }

        for (int key : classDistribution.keySet()) {
            p = (double) classDistribution.get(key) / totalSize;
            sum += p * p;
        }
        return 1 - sum;
    }

    public double weightedGini(final HashMap<Integer, ArrayList<Integer>> splits, final int[] labels, int nodeSize) {
        double wgini = 0.0;
        double gini;
        double splitSize;

        if (splits == null)
            return Double.POSITIVE_INFINITY;

        int totalSize = 0;
        for (int key : splits.keySet()) {
            if (splits.get(key) == null) {
                gini = 0;
                splitSize = 0;
            } else {
                gini = gini(splits.get(key), labels);
                splitSize = splits.get(key).size();
            }
            wgini = wgini + splitSize * gini;
            totalSize += splitSize;
        }
        return wgini / totalSize;
    }

    public String getClassName() {
        return this.getClass().getSimpleName();
    }

    public SplitterType getSplitterType() {
        return splitterType;
    }

    @Override
    public String toString() {
        return splitterType.toString();
    }

    public static boolean isValidSplit(final HashMap<Integer, ArrayList<Integer>> splits) {
        if (splits.size() < 2)
            return false;

        for (int key : splits.keySet())
            if (splits.get(key) == null || splits.get(key).size() == 0)
                return true;

        return false;
    }
}
