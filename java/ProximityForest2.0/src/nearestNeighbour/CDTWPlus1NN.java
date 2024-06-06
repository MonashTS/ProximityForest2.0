package nearestNeighbour;

import application.Application;
import datasets.Sequences;
import distances.CDTW_ge;
import results.PredictionResults;

public class CDTWPlus1NN extends CDTW1NN {
    protected double gammaExponent;
    protected double[] exponents = new double[]{1.0, 1.5, 0.666666, 2.0, 0.5};
    public String name = "CDTW+1NN";

    public CDTWPlus1NN() {
        this.classifierIdentifier = name;
        this.trainingOptions = OneNNTrainOpts.LOOCV;
    }

    public CDTWPlus1NN(final int paramId, final double exp) {
        super(paramId);
        this.bestParamId = paramId;
        this.gammaExponent = exp;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public CDTWPlus1NN(final int paramId, final double exp, final int useDerivative) {
        super(paramId, useDerivative);
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.gammaExponent = exp;
    }

    public CDTWPlus1NN(final Sequences xTrain) {
        super(xTrain);
        this.classifierIdentifier = name;
    }

    public CDTWPlus1NN(final int paramId, final double exp, final Sequences xTrain) {
        this(xTrain);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.gammaExponent = exp;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public CDTWPlus1NN(final Sequences xTrain, final String trainingOpts) {
        this(xTrain);
        this.trainingOptions = strToTrainOpts(trainingOpts);
    }

    public CDTWPlus1NN(final Sequences xTrain, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 1)
            this.classifierIdentifier = "d" + useDerivative + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.setTrainingData(xTrain);
        this.r = 1;
        this.window = xTrain.length();
    }

    public CDTWPlus1NN(final int paramId, final double exp, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.gammaExponent = exp;
        if (paramId >= 0) this.trainingOptions = OneNNTrainOpts.LOOCV0;
    }

    public CDTWPlus1NN(final Sequences xTrain, final String trainingOpts, final int useDerivative) {
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
                "\n[CLASSIFIER SUMMARY] gammaExponent: " + gammaExponent +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final double[] first, final double[] second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        window = getWindowSize(Math.max(first.length, second.length), r);
        return CDTW_ge.distance(first, second, window, cutOffValue, gammaExponent);
    }

    @Override
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
        return "\"warping_window\":" + this.window + ",\"r\":" + this.r + ",\"ge\":" + this.gammaExponent;
    }
}
