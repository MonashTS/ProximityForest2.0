package tree;

import datasets.Sequence;
import tree.results.NodeSplitterResult;
import tree.splitters.NodeSplitter;
import tree.splitters.PF2Splitter;
import tree.splitters.PFSplitter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import static tree.splitters.NodeSplitter.SplitterType.PF;
import static tree.splitters.NodeSplitter.SplitterType.PF2;
import static tree.splitters.NodeSplitter.isValidSplit;

public class Node {
    public transient Tree tree;
    public transient Node parent;
    public Random rand;

    public int nodeID;
    public int nodeDepth;
    public int nodeSize;

    public boolean isLeaf;
    public Integer label = null;
    public NodeSplitter bestSplitter;
    public double giniValue;
    public int[] indices;
    private HashMap<Integer, Node> children;

    protected HashMap<Integer, Integer> classDistribution;

    public Node(final Tree tree, final Node parent, final int nodeID) {
        this.tree = tree;
        this.parent = parent;
        this.nodeID = nodeID;
        this.isLeaf = false;

        if (parent == null) {
            this.nodeDepth = 0;
            this.rand = tree.rand;
        } else {
            this.nodeDepth = parent.nodeDepth + 1;
            this.rand = parent.rand;
        }
    }

    public void fit(final int[] indices) throws Exception {
        // indices should reference to train data in parent
        this.nodeSize = indices.length;
        this.indices = indices;

        getClassDistribution();

        this.giniValue = gini(this.classDistribution, this.nodeSize);

        // check stopping criteria
        if (stopBuilding()) return;

        HashMap<Integer, ArrayList<Integer>> bestSplit = fitSplitters();

        // if we cannot do a sensible split on the training data, convert the node to a leaf and return
        if (bestSplit == null || isValidSplit(bestSplit)) {
            makeLeaf();
            return;
        }

        // create child nodes first --  done separately from training loop for debugging reasons
        // (to set breakpoints and inspect all children before going into next level for training)
        this.children = new HashMap<>(bestSplit.size());
        for (int key : bestSplit.keySet())
            this.children.put(key, new Node(this.tree, this, this.tree.numNodes++));

        // train children using the best splitter
        for (int key : bestSplit.keySet()) {
            ArrayList<Integer> splitIdx = bestSplit.get(key);
            int[] childDataIndices = new int[splitIdx.size()];
            for (int i = 0; i < splitIdx.size(); i++)
                childDataIndices[i] = splitIdx.get(i);

            this.children.get(key).fit(childDataIndices);
        }

    }

    public HashMap<Integer, Integer> getClassDistribution() {
        if (this.classDistribution == null) {
            this.classDistribution = new HashMap<>();
            for (int i = 0; i < this.nodeSize; i++) {
                final int label = this.tree.getTrainLabel(this.indices[i]);

                if (classDistribution.containsKey(label))
                    classDistribution.put(label, classDistribution.get(label) + 1);
                else classDistribution.put(label, 1);
            }
        }

        return this.classDistribution;
    }

    public Integer getLabel() {
        return this.label;
    }

    public Node getChildren(final int key) {
        return children.get(key);
    }

    public int predict(final Sequence query) {
        final long startTime = System.nanoTime();
        final int label = bestSplitter.predict(query);
        final long stopTime = System.nanoTime();
        final NodeSplitter.SplitterType bestSplitterType = bestSplitter.getSplitterType();
        this.tree.splitterTime.put(bestSplitterType, this.tree.splitterTime.get(bestSplitterType) + stopTime - startTime);
        return label;
    }

    protected boolean stopBuilding() {
        // defensive case, to catch buggy splitters
        if (this.indices == null) {
            this.isLeaf = true;
            if (this.tree != null) this.tree.numLeaves++;
            return true;
        }

        // defensive case, if no data convert it to a leaf, use the label of test sample during prediction
        if (this.nodeSize == 0) {
            this.isLeaf = true;
            return true;
        }

        // if maxDepth == -1, then there is no limit on the tree depth
        if (this.tree.forest.maxDepth != -1 && this.nodeDepth >= this.tree.forest.maxDepth) {
            this.makeLeaf();
            return true;
        }

        if (this.giniValue <= this.tree.leafGiniThreshold) {
            this.makeLeaf();
            return true;
        }

        if (this.nodeSize <= this.tree.minNodeSize) {
            this.makeLeaf();
            return true;
        }

        return false;
    }

    protected void makeLeaf() {
        switch (this.tree.ensembleVotingScheme) {
            case majority:
                this.label = getMajorityClass();
                break;
            case prob:
                throw new RuntimeException("Not implemented yet");
            default:
                throw new RuntimeException("Unknown ensembleVotingScheme");
        }
        this.isLeaf = true;
        if (this.tree != null) {
            this.tree.numLeaves++;
            this.tree.depth = Math.max(this.nodeDepth, this.tree.depth);
            this.tree.leafCount++;
        }
    }

    protected int getMajorityClass() {
        final List<Integer> majorityClass = new ArrayList<>();
        int bsfCount = Integer.MIN_VALUE;

        for (int key : this.classDistribution.keySet()) {
            if (bsfCount < this.classDistribution.get(key)) {
                bsfCount = this.classDistribution.get(key);
                majorityClass.clear();
                majorityClass.add(key);
            } else if (bsfCount == this.classDistribution.get(key)) {
                majorityClass.add(key);
            }
        }

        final int r = rand.nextInt(majorityClass.size());
        return majorityClass.get(r);
    }

    public int getNumClasses() {
        if (this.classDistribution == null) this.getClassDistribution();

        return this.classDistribution.size();
    }

    protected double gini(final HashMap<Integer, Integer> classDistribution, final int nodeSize) {
        double sum = 0.0;
        double p;

        for (int key : classDistribution.keySet()) {
            p = (double) classDistribution.get(key) / nodeSize;
            sum += p * p;
        }

        return 1 - sum;
    }

    protected HashMap<Integer, ArrayList<Integer>> fitSplitters() throws Exception {
        double bestWeightedGini = Double.POSITIVE_INFINITY;

        NodeSplitterResult bestNodeSplitterResult;

        final List<NodeSplitter> splitters = new ArrayList<>();
        final List<NodeSplitterResult> splitterResults = new ArrayList<>();

        final HashMap<Integer, int[]> dataPerClass;
        if (this.tree.isPfEnabled() || this.tree.isPf2HEnabled())
            dataPerClass = getDataIndicesPerClass(this.tree.yTrain, this.tree.numClass, this.indices);
        else dataPerClass = null;

        // distance based
        if (this.tree.isPfEnabled()) fitPFSplitter(dataPerClass, splitters, splitterResults);
        if (this.tree.isPf2HEnabled()) fitPF2H3Splitter(dataPerClass, splitters, splitterResults);

        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < splitterResults.size(); i++) {
            final NodeSplitterResult result = splitterResults.get(i);
            if (result == null) continue;

            if (result.weightedGini < bestWeightedGini) {
                bestWeightedGini = result.weightedGini;
                candidates = new ArrayList<>();
                candidates.add(i);
            } else if (result.weightedGini == bestWeightedGini) {
                candidates.add(i);
            }
        }
        if (candidates.isEmpty())
            return null;

        int bestIndex = candidates.get(rand.nextInt(candidates.size()));
        bestSplitter = splitters.get(bestIndex);
        bestNodeSplitterResult = splitterResults.get(bestIndex);
        final NodeSplitter.SplitterType bestSplitterType = bestSplitter.getSplitterType();
        this.tree.splitterCount.put(bestSplitterType, this.tree.splitterCount.get(bestSplitterType) + 1);
        if (bestNodeSplitterResult == null) return null;

        return bestNodeSplitterResult.splits;
    }

    protected void fitPF2H3Splitter(final HashMap<Integer, int[]> dataPerClass, final List<NodeSplitter> splitters, final List<NodeSplitterResult> splitterResults) throws Exception {
        final long startTime = System.nanoTime();
        final int numSplits = this.tree.splittersPerNode.get(PF2);

        for (int i = 0; i < numSplits; i++) {
            final PF2Splitter currentSplitter = new PF2Splitter(this);
            final NodeSplitterResult currentSplitterResult = currentSplitter.fit(dataPerClass, this.indices);

            splitters.add(currentSplitter);
            splitterResults.add(currentSplitterResult);
        }
        this.tree.splitterTime.put(PF2, this.tree.splitterTime.get(PF2) + (System.nanoTime() - startTime));
    }

    protected void fitPFSplitter(final HashMap<Integer, int[]> dataPerClass, final List<NodeSplitter> splitters, final List<NodeSplitterResult> splitterResults) throws Exception {
        final long startTime = System.nanoTime();
        final int numSplits = this.tree.splittersPerNode.get(PF);

        for (int i = 0; i < numSplits; i++) {
            final PFSplitter currentSplitter = new PFSplitter(this);
            final NodeSplitterResult currentSplitterResult = currentSplitter.fit(dataPerClass, this.indices);

            splitters.add(currentSplitter);
            splitterResults.add(currentSplitterResult);
        }
        this.tree.splitterTime.put(PF, this.tree.splitterTime.get(PF) + (System.nanoTime() - startTime));
    }


    public HashMap<Integer, int[]> getDataIndicesPerClass(final int[] yTrain, final int numClass, final int[] indices) {
        final HashMap<Integer, int[]> split = new HashMap<>(numClass);
        final HashMap<Integer, Integer> splitCount = new HashMap<>(numClass);
        final HashMap<Integer, Integer> classDistribution = getClassDistribution();

        for (final int index : indices) {
            final int label = yTrain[index];
            if (!split.containsKey(label)) {
                final int numInstances = classDistribution.get(label);
                split.put(label, new int[numInstances]);
                splitCount.put(label, 0);
            }
            final int[] temp = split.get(label);
            temp[splitCount.get(label)] = index;
            split.put(label, temp);
            splitCount.put(label, splitCount.get(label) + 1);
        }
        return split;
    }
}

