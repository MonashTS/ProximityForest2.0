package nearestNeighbour;

import datasets.Sequences;
import distances.ADTW;

public class ADTW1NN extends OneNearestNeighbour {

    public static final int nSamples = 4000;
    public static final int exponent = 5;
    protected double weight;
    protected double maxWeight;
    protected double[] weights;
    protected boolean isWeightComputed = false;
    public String name = "ADTW1NN";

    public ADTW1NN() {
        this.classifierIdentifier = name;
        this.trainingOptions = OneNNTrainOpts.LOOCV;
    }

    public ADTW1NN(final int paramId) {
        this();
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ADTW1NN(final int paramId, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ADTW1NN(final Sequences xTrain) {
        this();
        this.classifierIdentifier = name;
        this.setTrainingData(xTrain);
    }

    public ADTW1NN(final int paramId, final Sequences xTrain) {
        this(xTrain);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ADTW1NN(final Sequences xTrain, final String trainingOpts) {
        this(xTrain);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    public ADTW1NN(final Sequences xTrain, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 1)
            this.classifierIdentifier = "d" + useDerivative + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.setTrainingData(xTrain);
    }

    public ADTW1NN(final int paramId, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ADTW1NN(final Sequences xTrain, final String trainingOpts, final int useDerivative) {
        this(xTrain, useDerivative);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] nThread: " + nThreads +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] weight: " + weight +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final double[] first, final double[] second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        return ADTW.distance(first, second, weight, cutOffValue);
    }

    public double distance(final double[] first, final double[] second, final double weight, final double cutOffValue) {
        return ADTW.distance(first, second, weight, cutOffValue);
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        initWeights();
        weight = weights[paramId];
    }

    public void initWeights() {
        weights = xTrain.initADTWWeights(this.nSamples, nParams, this.exponent);
        isWeightComputed = true;
    }

    @Override
    public String getParamInformationString() {
        return "\"omega\":" + this.weight + ",\"max_weight\":" + maxWeight;
    }
}
