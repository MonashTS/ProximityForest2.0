package nearestNeighbour;

import application.Application;
import datasets.Sequences;
import distances.ADTW_ge;
import results.PredictionResults;

public class ADTWPlus1NN extends ADTW1NN {
    protected double gammaExponent;
    protected double[] exponents = new double[]{1.0, 2.0, 0.5};
    public String name = "ADTW+1NN";

    public ADTWPlus1NN() {
        this.classifierIdentifier = name;
        this.trainingOptions = OneNNTrainOpts.LOOCV;
    }

    public ADTWPlus1NN(final int paramId, final double exp) {
        super(paramId);
        this.bestParamId = paramId;
        this.gammaExponent = exp;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ADTWPlus1NN(final int paramId, final double exp, final int useDerivative) {
        super(paramId, useDerivative);
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.gammaExponent = exp;
    }

    public ADTWPlus1NN(final Sequences xTrain) {
        super(xTrain);
        this.classifierIdentifier = name;
    }

    public ADTWPlus1NN(final int paramId, final double exp, final Sequences xTrain) {
        this(xTrain);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.gammaExponent = exp;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ADTWPlus1NN(final Sequences xTrain, final String trainingOpts) {
        this(xTrain);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    public ADTWPlus1NN(final Sequences xTrain, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 1)
            this.classifierIdentifier = "d" + useDerivative + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.setTrainingData(xTrain);
    }

    public ADTWPlus1NN(final int paramId, final double exp, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.gammaExponent = exp;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public ADTWPlus1NN(final Sequences xTrain, final String trainingOpts, final int useDerivative) {
        this(xTrain, useDerivative);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] nThread: " + nThreads +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] weight: " + weight +
                "\n[CLASSIFIER SUMMARY] gammaExponent: " + gammaExponent +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final double[] first, final double[] second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        return ADTW_ge.distance(first, second, weight[0], cutOffValue, gammaExponent);
    }

    public double distance(final double[] first, final double[] second, final double weight, final double cutOffValue) {
        return ADTW_ge.distance(first, second, weight, cutOffValue, gammaExponent);
    }

    public PredictionResults loocv(final Sequences xTrain) throws Exception {
        PredictionResults bestResults = new PredictionResults();
        if (gammaExponent > 0) exponents = new double[]{gammaExponent};
        int[] cvParams = new int[nParams * exponents.length];
        double[] cvAcc = new double[nParams * exponents.length];
        bestParamId = -1;
        int bestGe = 0;

        if (Application.verbose >= 0)
            System.out.println("[1-NN] LOOCV for " + this.classifierIdentifier + ", training ");

        final long start = System.nanoTime();

        for (int expId = 0; expId < exponents.length; expId++) {
            gammaExponent = exponents[expId];
            this.isWeightComputed = false;
            if (Application.verbose >= 1)
                System.out.print(gammaExponent + "loocv_acc = [");
            else if (Application.verbose >= 0)
                System.out.print(gammaExponent + "");
            for (int paramId = 0; paramId < nParams; paramId++) {
                final int idx = paramId + expId * nParams;
                cvParams[idx] = paramId;
                if (Application.verbose >= 0)
                    System.out.print(".");
                PredictionResults loocvResults = loocvAccAndPreds(xTrain, paramId);
                if (Application.verbose >= 1)
                    System.out.print(loocvResults.accuracy + ",");
                cvAcc[idx] = loocvResults.accuracy;

                if (loocvResults.accuracy > bestResults.accuracy) {
                    bestParamId = paramId;
                    bestResults = loocvResults;
                    bestGe = expId;
                }
            }
            if (Application.verbose >= 1)
                System.out.println("];");
            else if (Application.verbose >= 0)
                System.out.println();
        }
        final long end = System.nanoTime();

        trainingTime = 1.0 * (end - start) / 1e9;

        bestResults.setTime(start, end);
        bestResults.setParamId(bestParamId + bestGe * nParams);
        bestResults.setCvAcc(cvAcc);
        bestResults.setCvParams(cvParams);
        bestResults.setTrainTest("train");

        this.setTrainingData(xTrain);
        this.setParamsFromParamId(bestParamId);
        gammaExponent = exponents[bestGe];
        if (Application.verbose >= 0)
            System.out.printf("[1-NN] LOOCV Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bestResults.accuracy,
                    bestResults.doTime());

        isTrain = true;

        return bestResults;
    }

    @Override
    public String getParamInformationString() {
        return "\"omega\":" + this.weight + ",\"ge\":" + this.gammaExponent;
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        initWeights(this.gammaExponent);

        weight = weights[paramId];
    }

    public void initWeights(final double gammaExponent) {
        weights = xTrain.initADTWWeights(this.nSamples, nParams, this.exponent, gammaExponent);
        isWeightComputed = true;
    }
}


