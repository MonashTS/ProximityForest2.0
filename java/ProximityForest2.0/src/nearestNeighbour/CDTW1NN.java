package nearestNeighbour;

import datasets.Sequences;
import distances.ADTW;
import distances.CDTW;
import transforms.DerivativeFilter;

import java.util.Random;

public class CDTW1NN extends OneNearestNeighbour {
    protected double r;
    protected int window;
    final int ubParam = 0;
    public String name = "CDTW1NN";

    public CDTW1NN() {
        this.classifierIdentifier = name;
        this.trainingOptions = OneNNTrainOpts.LOOCV;
    }

    public CDTW1NN(final int paramId) {
        this();
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public CDTW1NN(final int paramId, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public CDTW1NN(final Sequences xTrain) {
        this();
        this.classifierIdentifier = name;
        this.setTrainingData(xTrain);
        this.r = 1;
        this.window = xTrain.length();
    }

    public CDTW1NN(final int paramId, final Sequences xTrain) {
        this(xTrain);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public CDTW1NN(final Sequences xTrain, final String trainingOpts) {
        this(xTrain);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    public CDTW1NN(final Sequences xTrain, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 1)
            this.classifierIdentifier = "d" + useDerivative + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.setTrainingData(xTrain);
        this.r = 1;
        this.window = xTrain.length();
    }

    public CDTW1NN(final int paramId, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public CDTW1NN(final Sequences xTrain, final String trainingOpts, final int useDerivative) {
        this(xTrain, useDerivative);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] nThread: " + nThreads +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] r: " + r +
                "\n[CLASSIFIER SUMMARY] window: " + window +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final double[] first, final double[] second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        window = getWindowSize(Math.max(first.length, second.length), r);
        return CDTW.distance(first, second, window, cutOffValue);
    }

    @Override
    public double distance(final double[][] first, final double[][] second, final double cutOffValue) {
        double dist = 0;
        for (int i = 0; i < first.length; i++){
            dist += CDTW.distance(first[i], second[i], window, cutOffValue);
        }
        return dist;
    }

    @Override
    public void setTrainingData(final Sequences xTrain) {
        this.xTrain = xTrain;
        if (!this.derComplete)
            for (int i = 0; i < this.useDerivative; i++)
                this.xTrain = DerivativeFilter.getFirstDerivative(this.xTrain);
        this.window = getWindowSize(this.xTrain.length(), this.r);

        this.numClass = this.xTrain.getNumClasses();
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;
        r = 1.0 * paramId / 100;
        window = getWindowSize(this.xTrain.length(), this.r);
    }


    @Override
    public void setRandomParams(Random rand) {
        this.window = rand.nextInt((xTrain.length() + 1) / 4);
        this.r = 1.0 * this.window / xTrain.length();
    }

    @Override
    public String getParamInformationString() {
        return "\"warping_window\":" + this.window + ",\"r\":" + this.r;
    }

    public int getWindowSize(final int n, final double r) {
        return (int) Math.ceil(r * n);
    }


}
