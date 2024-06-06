package tree;

import application.Application;
import datasets.Sequence;
import datasets.Sequences;
import results.PredictionResults;
import transforms.DerivativeFilter;
import transforms.Transforms;
import utils.Tools;

import java.util.HashMap;

import static transforms.Transforms.TimeSeriesTransforms.raw;

/**
 * An abstract class for any time series classifier
 */
public abstract class TimeSeriesClassifier extends AbstractClassifier {
    public enum OneNNTrainOpts {
        LOOCV0, LOOCV,
    }


    // variables
    public HashMap<Transforms.TimeSeriesTransforms, Sequences> trainDataset;
    public HashMap<Transforms.TimeSeriesTransforms, Sequences> testDataset;
    public Sequences xTrain;
    public Sequences xTest;
    public int useDerivative = 0;
    public boolean derComplete = false;

    public OneNNTrainOpts trainingOptions = OneNNTrainOpts.LOOCV;

    /**
     * Summary of the model
     */
    public void summary() {
        System.out.println(this);
    }

    /**
     * Abstract fit function
     *
     * @param xTrain training examples
     * @return training prediction results
     */
    public abstract PredictionResults fit(final Sequences xTrain) throws Exception;

    /**
     * Abstract partial fit function
     *
     * @param xTrain  training examples
     * @param nSample number of samples for training
     * @return training prediction results
     */
    public abstract PredictionResults fit(final Sequences xTrain, final double nSample) throws Exception;

    /**
     * Given an instance x, predict its class
     *
     * @param x instance to predict
     */
    public abstract int[] predict(final Sequence x) throws Exception;

    /**
     * Evaluate a set of examples
     *
     * @param xTest test examples
     * @return test prediction results
     */
    public PredictionResults evaluate(final Sequences xTest) throws Exception {
        final int testSize = xTest.size();
        int numClass = this.numClass;

        testClassCounts = new int[testSize][numClass];
        testPreds = new int[testSize];
        testCorrect = 0;

        final long startTime = System.nanoTime();

        for (int i = 0; i < testSize; i++) {
            final int[] a = predict(xTest.get(i));
            final int predictClass = a[0];
            System.arraycopy(a, 1, testClassCounts[i], 0, numClass);
            if (predictClass == xTest.get(i).classLabel) testCorrect++;
            testPreds[i] = predictClass;
        }

        final long stopTime = System.nanoTime();

        final double acc = 1.0 * testCorrect / testSize;
        testResults = new PredictionResults(testCorrect, acc, testPreds, testClassCounts, "test");
        testResults.setTime(startTime, stopTime);

        return testResults;
    }

    /**
     * Get parameter information as string
     *
     * @return parameter information
     */
    public String getParamInformationString() {
        return "";
    }

    /**
     * Convert string to training options
     *
     * @param str string to convert
     * @return training options
     */
    public OneNNTrainOpts strToTrainOpts(final String str) {
        switch (str.toLowerCase()) {
            case "loocv":
                return OneNNTrainOpts.LOOCV;
            default:
                return OneNNTrainOpts.LOOCV0;
        }
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier + "\n[CLASSIFIER SUMMARY] nThread: " + nThreads + "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions;
    }


    // --- --- --- Set functions
    public void setTrainingData(final Sequences xTrain) {
        this.xTrain = xTrain;
        if (!this.derComplete)
            for (int i = 0; i < this.useDerivative; i++) {
                this.xTrain = DerivativeFilter.getFirstDerivative(this.xTrain);
                this.derComplete = true;
            }

        this.numClass = xTrain.getNumClasses();
        this.trainIndices = Tools.arange(this.xTrain.size());
    }

    public void setTrainingData(final HashMap<Transforms.TimeSeriesTransforms, Sequences> xTrain) {
        this.xTrain = xTrain.get(raw);
        if (!this.derComplete) {
            for (int i = 0; i < this.useDerivative; i++) {
                Transforms.TimeSeriesTransforms key = raw;
                switch (i) {
                    case 0:
                        key = Transforms.TimeSeriesTransforms.d1;
                        break;
                    case 1:
                        key = Transforms.TimeSeriesTransforms.d2;
                        break;
                }
                if (!xTrain.containsKey(key)) {
                    this.xTrain = DerivativeFilter.getFirstDerivative(this.xTrain);
                    xTrain.put(key, this.xTrain);
                } else this.xTrain = xTrain.get(key);
                this.derComplete = true;
            }
        }

        this.trainDataset = xTrain;
        this.numClass = this.trainDataset.get(raw).getNumClasses();
    }

    public void setTestData(final Sequences xTest) {
        this.xTest = xTest;
    }

    public void setThreads(int threads) {
        // make sure n_threads is less than n_cpu
        if (threads < 0) threads = Runtime.getRuntime().availableProcessors();
        else threads = Math.min(threads, Runtime.getRuntime().availableProcessors());
        threads = Math.max(threads, 1);

        if (Application.verbose > 0) System.out.println("Setting nThreads to " + threads);
        this.nThreads = threads;
    }

    public void setVerbose(int verbose) {
        Application.verbose = verbose;
    }
}
