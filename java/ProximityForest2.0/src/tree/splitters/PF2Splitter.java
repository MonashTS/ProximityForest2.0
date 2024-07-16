package tree.splitters;

import application.Application;
import datasets.Sequence;
import nearestNeighbour.OneNearestNeighbour;
import transforms.DerivativeFilter;
import tree.Node;

import static transforms.Transforms.TimeSeriesTransforms.*;

public class PF2Splitter extends PFSplitter {
    public static String[] enabledDistanceTransform = new String[]{"", "d1"};
    public static double[] enabledDistancesCostFunction = new double[]{0.5, 1.0, 2.0};

    public PF2Splitter(final Node node) {
        this.node = node;
        if (this.node != null) this.rand = Application.rand;
        else rand = this.node.rand;

        name = "PF2";
        splitterType = SplitterType.PF2;
        splitterClass = getClassName();

        enabledDistances = new String[]{"ADTW1NN", "CDTW1NN", "LCSS1NN", "ED1NN"};
        // initialise the measure here.
        int r = rand.nextInt(enabledDistances.length);
        String selectedDistance = enabledDistances[r];
        transform = raw;

        if (selectedDistance.equals("ED1NN")) {
            transform = hydra;

            r = rand.nextInt(enabledDistancesCostFunction.length);
            if (enabledDistancesCostFunction[r] == 2)
                selectedDistance = "ED1NN";
            else
                selectedDistance = "Minkowski1NN-" + enabledDistancesCostFunction[r];
        } else {
            // select the transform
            r = rand.nextInt(enabledDistanceTransform.length);
            String selectedTransform = enabledDistanceTransform[r];
            if (!selectedTransform.isEmpty()) {
                transform = d1;
                selectedTransform = selectedTransform + "-";
            }

            String selectedCost = "";
            if (!selectedDistance.equals("LCSS1NN")) {
                // select cost function if not LCSS
                r = rand.nextInt(enabledDistancesCostFunction.length);
                if (enabledDistancesCostFunction[r] < 2)
                    selectedCost = "-" + enabledDistancesCostFunction[r];
            }
            selectedDistance = selectedTransform + selectedDistance + selectedCost;
        }

        if (!this.node.tree.distanceCount.containsKey(selectedDistance))
            this.node.tree.distanceCount.put(selectedDistance, 0);
        this.node.tree.distanceCount.put(selectedDistance, this.node.tree.distanceCount.get(selectedDistance) + 1);

        measure = OneNearestNeighbour.init(
                this.node.tree.getTrainTS(), selectedDistance
        );
        if (transform == d1) measure.derComplete = true;
        assert node != null;
        measure.setTrainingData(node.tree.trainDataset.get(transform));

        measure.setRandomParams(this.rand);
    }

    public int predict(final Sequence query) {
        if (transform == hydra) {
            final double[] q = query.transforms.get(hydra);
            return findNearestExemplar(q, measure, exemplars);
        } else if (measure.useDerivative > 0) {
            final double[][] q = DerivativeFilter.getFirstDerivative(query.data);
            return findNearestExemplar(q, measure, exemplars);
        }
        return findNearestExemplar(query.data, measure, exemplars);
    }
}
