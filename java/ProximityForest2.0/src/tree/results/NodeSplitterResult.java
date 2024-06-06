package tree.results;


import tree.splitters.NodeSplitter;

import java.util.ArrayList;
import java.util.HashMap;

public class NodeSplitterResult implements Comparable<NodeSplitterResult> {

    public NodeSplitter splitter;
    public HashMap<Integer, ArrayList<Integer>> splits;
    public HashMap<Integer, ArrayList<Integer>> splitIndices;
    public Double weightedGini;

    public NodeSplitterResult(final NodeSplitter splitter, final int numChildren) {
        this.splitter = splitter;
        this.splits = new HashMap<>(numChildren);
        this.splitIndices = new HashMap<>(numChildren);
        this.weightedGini = Double.POSITIVE_INFINITY;
    }

    @Override
    public int compareTo(NodeSplitterResult o) {
        return o.weightedGini.compareTo(weightedGini);
    }

    @Override
    public String toString() {
        return weightedGini + "";
    }
}
