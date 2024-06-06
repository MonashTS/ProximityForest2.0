package nearestNeighbour;

import datasets.Sequences;
import distances.ED;

public class ED1NN extends OneNearestNeighbour {
    public String name = "ED1NN";

    public ED1NN() {
        this.classifierIdentifier = name;
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ED1NN(final int paramId) {
        this();
        this.bestParamId = paramId;
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ED1NN(final int paramId, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.bestParamId = paramId;
    }

    public ED1NN(final Sequences xTrain) {
        this();
        this.classifierIdentifier = name;
        this.setTrainingData(xTrain);
    }

    public ED1NN(final int paramId, final Sequences xTrain) {
        this(xTrain);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
    }

    public ED1NN(final Sequences xTrain, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 1)
            this.classifierIdentifier = "d" + useDerivative + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.setTrainingData(xTrain);
    }

    public ED1NN(final int paramId, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
    }

    @Override
    public double distance(final double[] first, final double[] second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        return ED.distance(first, second, cutOffValue);
    }

    @Override
    public OneNNTrainOpts strToTrainOpts(final String str) {
        return OneNNTrainOpts.LOOCV0;
    }
}
