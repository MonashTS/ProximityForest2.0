package nearestNeighbour;

import datasets.Sequences;
import distances.ED;
import distances.LCSS;
import utils.Tools;

import java.util.Random;

public class LCSS1NN extends OneNearestNeighbour {
    protected int delta;
    protected double epsilon;
    protected int[] deltas;
    protected double[] epsilons;
    protected boolean epsilonsAndDeltasRefreshed;
    public String name = "LCSS1NN";

    public LCSS1NN() {
        this.classifierIdentifier = name;
        this.trainingOptions = OneNNTrainOpts.LOOCV;
    }

    public LCSS1NN(final int paramId) {
        this();
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public LCSS1NN(final int paramId, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public LCSS1NN(final Sequences xTrain) {
        this();
        this.classifierIdentifier = name;
        this.setTrainingData(xTrain);
    }

    public LCSS1NN(final int paramId, final Sequences xTrain) {
        this(xTrain);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public LCSS1NN(final Sequences xTrain, final String trainingOpts) {
        this(xTrain);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    public LCSS1NN(final Sequences xTrain, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 1)
            this.classifierIdentifier = "d" + useDerivative + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.setTrainingData(xTrain);
    }

    public LCSS1NN(final int paramId, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public LCSS1NN(final Sequences xTrain, final String trainingOpts, final int useDerivative) {
        this(xTrain, useDerivative);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] nThread: " + nThreads +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] delta: " + delta +
                "\n[CLASSIFIER SUMMARY] epsilon: " + epsilon +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }


    @Override
    public double distance(final double[] first, final double[] second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        return LCSS.distance(first, second, epsilon, delta, cutOffValue);
    }

    @Override
    public double distance(final double[][] first, final double[][] second, final double cutOffValue) {
        double dist = 0;
        for (int i = 0; i < first.length; i++){
            dist += LCSS.distance(first[i], second[i], epsilon, delta, cutOffValue);
        }
        return dist;
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        if (!epsilonsAndDeltasRefreshed) {
            double stdTrain = xTrain.getStd_p();
            double stdFloor = stdTrain * 0.2;
            epsilons = Tools.getInclusive10(stdFloor, stdTrain);
            deltas = Tools.getInclusive10(0, (xTrain.length() + 1) / 4);
            epsilonsAndDeltasRefreshed = true;
        }
        this.delta = deltas[paramId % 10];
        this.epsilon = epsilons[paramId / 10];
    }

    @Override
    public void setRandomParams(Random rand) {
        final double stdTrain = xTrain.getStd_p();
        final double stdFloor = stdTrain * 0.2;
        this.epsilon = stdFloor + (stdTrain - stdFloor) * rand.nextDouble();
        this.delta = rand.nextInt((xTrain.length() + 1) / 4);
    }

    @Override
    public String getParamInformationString() {
        return "\"delta\":" + this.delta + ",\"epsilon\":" + this.epsilon;
    }

}
