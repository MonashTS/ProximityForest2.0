package nearestNeighbour;

import application.Application;
import datasets.Sequences;
import distances.Minkowski;
import results.PredictionResults;

public class Minkowski1NN extends OneNearestNeighbour {
    protected double[] exponents = new double[]{1.0, 1.5, 0.666666, 2.0, 0.5};
    protected double p = 2;

    public String name = "Minkowski1NN";

    public Minkowski1NN() {
        this.classifierIdentifier = name;
        this.trainingOptions = OneNNTrainOpts.LOOCV;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final int paramId) {
        this();
        this.bestParamId = paramId;
        this.setParamsFromParamId(paramId);
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final double p) {
        this();
        this.p = p;
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final int paramId, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.bestParamId = paramId;
        this.setParamsFromParamId(paramId);
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final double p, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 0)
            this.classifierIdentifier = "d" + useDerivative + "-" + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.p = p;
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final Sequences xTrain) {
        this();
        this.classifierIdentifier = name;
        this.setTrainingData(xTrain);
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final int paramId, final Sequences xTrain) {
        this(xTrain);
        this.bestParamId = paramId;
        this.setParamsFromParamId(paramId);
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final double p, final Sequences xTrain) {
        this(xTrain);
        this.p = p;
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final Sequences xTrain, final String trainingOpts) {
        this(xTrain);
        this.trainingOptions = strToTrainOpts(trainingOpts);
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final Sequences xTrain, final int useDerivative) {
        this();
        this.classifierIdentifier = name;
        if (useDerivative > 1)
            this.classifierIdentifier = "d" + useDerivative + this.classifierIdentifier;
        this.useDerivative = useDerivative;
        this.setTrainingData(xTrain);
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final int paramId, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final double p, final Sequences xTrain, final int useDerivative) {
        this(xTrain, useDerivative);
        this.p = p;
        this.trainingOptions = OneNNTrainOpts.LOOCV0;
        this.nParams = exponents.length;
    }

    public Minkowski1NN(final Sequences xTrain, final String trainingOpts, final int useDerivative) {
        this(xTrain, useDerivative);
        this.trainingOptions = strToTrainOpts(trainingOpts);
        this.nParams = exponents.length;
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] nThread: " + nThreads +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] p: " + p +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public PredictionResults loocv(final Sequences xTrain) throws Exception {
        PredictionResults bestResults = new PredictionResults();

        final int nParams = exponents.length;

        int[] cvParams = new int[nParams];
        double[] cvAcc = new double[nParams];
        bestParamId = -1;

        if (Application.verbose > 0)
            System.out.print("[1-NN] LOOCV for " + this.classifierIdentifier + ", training ");
        if (Application.verbose >= 1)
            System.out.print("loocv_acc = [");

        final long start = System.nanoTime();

        for (int paramId = 0; paramId < nParams; paramId++) {
            cvParams[paramId] = paramId;
            if (Application.verbose > 0)
                System.out.print(".");
            PredictionResults loocvResults = loocvAccAndPreds(xTrain, paramId);
            if (Application.verbose >= 1)
                System.out.print(loocvResults.accuracy + ",");
            cvAcc[paramId] = loocvResults.accuracy;

            if (loocvResults.accuracy > bestResults.accuracy) {
                bestParamId = paramId;
                bestResults = loocvResults;
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose >= 1)
            System.out.println("];");
        else if (Application.verbose > 0)
            System.out.println();
        trainingTime = 1.0 * (end - start) / 1e9;

        bestResults.setTime(start, end);
        bestResults.setParamId(bestParamId);
        bestResults.setCvAcc(cvAcc);
        bestResults.setCvParams(cvParams);
        bestResults.setTrainTest("train");

        this.setTrainingData(xTrain);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 0)
            System.out.printf("[1-NN] LOOCV Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bestResults.accuracy,
                    bestResults.doTime());

        isTrain = true;

        return bestResults;
    }

    @Override
    public double distance(final double[] first, final double[] second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final double[] first, final double[] second, final double cutOffValue) {
        return Minkowski.distance(first, second, cutOffValue, p);
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (p > 0) return;
        if (paramId < 0) return;
        p = exponents[paramId];
    }

    @Override
    public String getParamInformationString() {
        return "\"p\":" + this.p;
    }

    @Override
    public OneNNTrainOpts strToTrainOpts(final String str) {
        switch (str.toLowerCase()) {
            case "loocv0":
                return OneNNTrainOpts.LOOCV0;
            default:
                return OneNNTrainOpts.LOOCV;
        }
    }
}
