package tree.splitters;

import datasets.Sequence;
import nearestNeighbour.OneNearestNeighbour;
import transforms.DerivativeFilter;
import transforms.Transforms;
import tree.Node;
import tree.results.NodeSplitterResult;

import java.util.ArrayList;
import java.util.HashMap;

import static transforms.Transforms.TimeSeriesTransforms.*;


public class PFSplitter extends NodeSplitter {
    protected OneNearestNeighbour measure;
    protected Transforms.TimeSeriesTransforms transform = raw;

    public static String[] enabledDistances = new String[]{"ED1NN", "DTW1NN", "CDTW1NN", "d1-DTW1NN", "d1-CDTW1NN", "WDTW1NN", "d1-WDTW1NN", "MSM1NN", "TWE1NN", "ERP1NN", "LCSS1NN"};

    public PFSplitter() {

    }

    public PFSplitter(final Node node) {
        super(node);
        name = "PF";
        splitterType = SplitterType.PF;
        splitterClass = getClassName();

        // initialise the measure here.
        final int r = rand.nextInt(enabledDistances.length);
        final String selectedDistance = enabledDistances[r];

        measure = OneNearestNeighbour.init(this.node.tree.getTrainTS(), selectedDistance);
        switch (measure.useDerivative) {
            case 1:
                transform = d1;
                break;
            case 2:
                transform = d2;
                break;
            default:
                transform = raw;
        }
        if (measure.useDerivative > 0) measure.derComplete = true;
        measure.setTrainingData(node.tree.trainDataset.get(transform));
        measure.setParamsFromParamId(rand.nextInt(measure.nParams));
    }

    public NodeSplitterResult fit(final HashMap<Integer, int[]> dataPerClass, final int[] nodeIndices) {
        exemplars = new HashMap<>(this.node.getNumClasses());

        for (int key : dataPerClass.keySet()) {
            int[] indicesThisClass = dataPerClass.get(key);
            int r = rand.nextInt(indicesThisClass.length);
            exemplars.put(key, indicesThisClass[r]);
        }

        return split(nodeIndices);
    }


    @Override
    public NodeSplitterResult split(final int[] nodeIndices) {
        final NodeSplitterResult result = new NodeSplitterResult(this, this.node.getNumClasses());
        int closestBranch;

        for (final int ix : nodeIndices) {
            closestBranch = findNearestExemplar(this.node.tree.getTrainSeries(ix, this.transform).data, measure, exemplars);

            if (!result.splits.containsKey(closestBranch)) {
                ArrayList<Integer> temp = new ArrayList<>(this.node.getClassDistribution().get(closestBranch));
                result.splits.put(closestBranch, temp);
            }

            result.splits.get(closestBranch).add(ix);
        }

        result.weightedGini = weightedGini(result.splits, this.node.tree.getTrainLabels(), this.node.nodeSize);

        this.isTrain = true;
        return result;
    }


    @Override
    public int predict(final Sequence query) {
        if (measure.useDerivative > 0) {
            final double[][] q = DerivativeFilter.getFirstDerivative(query.data);
            return findNearestExemplar(q, measure, exemplars);
        }
        return findNearestExemplar(query.data, measure, exemplars);
    }

    protected synchronized int findNearestExemplar(final double[] query, final OneNearestNeighbour oneNN, final HashMap<Integer, Integer> exemplars) {
        closestNodes.clear();
        double bsf = Double.POSITIVE_INFINITY;
        double dist;

        for (int key : exemplars.keySet()) {
            final int i = exemplars.get(key);
            final double[] exemplar = this.node.tree.getTrainSeries(i, this.transform).firstChannel();

            if (exemplar == query) return key;

            dist = oneNN.distance(query, exemplar, bsf);
            if (dist < bsf) {
                bsf = dist;
                closestNodes.clear();
                closestNodes.add(key);
            } else if (dist == bsf) {
                closestNodes.add(key);
            }
        }

        final int r = rand.nextInt(closestNodes.size());
        return closestNodes.get(r);
    }

    protected synchronized int findNearestExemplar(final double[][] query, final OneNearestNeighbour oneNN, final HashMap<Integer, Integer> exemplars) {
        closestNodes.clear();
        double bsf = Double.POSITIVE_INFINITY;
        double dist;

        for (int key : exemplars.keySet()) {
            final int i = exemplars.get(key);
            final double[][] exemplar = this.node.tree.getTrainSeries(i, this.transform).data;

            if (exemplar == query) return key;

            dist = oneNN.distance(query, exemplar, bsf);
            if (dist < bsf) {
                bsf = dist;
                closestNodes.clear();
                closestNodes.add(key);
            } else if (dist == bsf) {
                closestNodes.add(key);
            }
        }

        final int r = rand.nextInt(closestNodes.size());
        return closestNodes.get(r);
    }

    public String toString() {
        if (measure == null) return name + "[untrained]";
        else return name + "[" + measure.toString() + "]";
    }
}
