package nearestNeighbour;

import application.Application;
import datasets.Sequence;
import datasets.Sequences;
import results.PredictionResults;
import transforms.DerivativeFilter;
import transforms.Transforms;
import tree.TimeSeriesClassifier;
import utils.MultiThreadedTask;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

/**
 * An abstract class for one nearest neighbour
 */
public abstract class OneNearestNeighbour extends TimeSeriesClassifier {
    protected class OneNNLooCVTask implements Callable<Integer> {
        protected int[] index; // index (rows of the NN table)
        protected int[] stratifiedIndex; // stores stratified index, mapped from index
        protected int threadId; // thread id

        public OneNNLooCVTask(int[] index, int threadId) {
            this.index = index;
            this.threadId = threadId;
        }

        public OneNNLooCVTask(int[] index, int[] stratifiedIndex, int threadId) {
            this.index = index;
            this.stratifiedIndex = stratifiedIndex;
            this.threadId = threadId;
        }

        @Override
        public Integer call() throws Exception {
            if (Application.verbose > 0)
                System.out.println("[Thread " + threadId + "] Doing " + this.index.length + " -- " + this.index[0]);
            if (this.stratifiedIndex == null || this.stratifiedIndex.length == 0)
                for (int actualIndex : this.index) {
                    final Sequence query = xTrain.get(actualIndex);
                    final int[] a = classifyLoocv(query, actualIndex);
                    System.arraycopy(a, 1, trainClassCounts[actualIndex], 0, numClass);
                    trainPreds[actualIndex] = a[0];
                    if (trainPreds[actualIndex] == query.classLabel) trainCorrect++;
                }
            else if (this.index != null)
                for (int i : this.index) {
                    final int actualIndex = this.stratifiedIndex[i];
                    final Sequence query = xTrain.get(actualIndex);
                    final int[] a = classifyLoocv(query, actualIndex);
                    System.arraycopy(a, 1, trainClassCounts[i], 0, numClass);
                    trainPreds[i] = a[0];
                    if (trainPreds[i] == query.classLabel) trainCorrect++;
                }
            if (Application.verbose > 0)
                System.out.println("[Thread " + threadId + "] Complete ");
            return null;
        }
    }

    /**
     * A task for parallelising 1NN evaluation task
     */
    protected class OneNNEvaluateTask extends EvaluateTask {

        public OneNNEvaluateTask(int[] index, int threadCount) {
            super(index, threadCount);
        }

        @Override
        public Integer call() throws Exception {
            if (Application.verbose > 0)
                System.out.println("[Thread " + threadId + "] Evaluating " + this.index.length + " series");

            for (final int actualIndex : this.index) {
                final int[] a = predict(xTest.get(actualIndex));
                final int predictClass = a[0];
                synchronized (testClassCounts) {
                    System.arraycopy(a, 1, testClassCounts[actualIndex], 0, numClass);
                }
                if (predictClass == xTest.get(actualIndex).classLabel) testCorrect++;
                synchronized (testPreds) {
                    testPreds[actualIndex] = predictClass;
                }
            }
            if (Application.verbose > 0)
                System.out.println("[Thread " + threadId + "] Complete ");
            return null;
        }
    }

    public int nParams = 100;    // number of parameters

    protected int[][][] classCounts;          // class proba

    public int bestParamId;         // best parameter id
    public String bestParamStr;     // best parameter in string
    protected int paramId;          // parameter id

    protected int numSeries;
    protected int[] shuffledIndices;

    // --- --- --- distances functions

    public abstract double distance(final double[] first, final double[] second);

    public abstract double distance(final double[] first,
                                    final double[] second,
                                    final double cutOffValue);

    public abstract double distance(final double[][] first,
                                    final double[][] second,
                                    final double cutOffValue);

    /**
     * 1NN fit function
     *
     * @param xTrain training examples
     * @return training prediction results
     */
    @Override
    public PredictionResults fit(final Sequences xTrain) throws Exception {
        this.setTrainingData(xTrain);

        if (this.trainingOptions == OneNNTrainOpts.LOOCV)
            trainResults = loocv(this.xTrain);
        else
            trainResults = loocv0(this.xTrain);

        bestParamStr = getParamInformationString();
        trainResults.setParamStr(bestParamStr);

        return trainResults;
    }

    /**
     * 1NN partial fit function
     *
     * @param xTrain  training examples
     * @param nSample number of training samples
     * @return training prediction results
     */

    @Override
    public PredictionResults fit(final Sequences xTrain, double nSample) throws Exception {
        this.setTrainingData(xTrain);

        if (this.trainingOptions == OneNNTrainOpts.LOOCV)
            trainResults = loocv(nSample, this.xTrain);
        else
            trainResults = loocv0(nSample, this.xTrain);

        bestParamStr = getParamInformationString();
        trainResults.setParamStr(bestParamStr);

        return trainResults;
    }


    /**
     * Predict the class of a query
     *
     * @param query query time series
     * @return array where a[0] = class prediction and the rest is class proba
     */
    @Override
    public int[] predict(final Sequence query) {
        int[] classCounts = new int[this.numClass];

        double dist;
        double[] querySeries = query.firstChannel();
        Sequence candidate = this.xTrain.get(0);
        double bsfDistance = distance(querySeries, candidate.firstChannel());
        classCounts[candidate.classLabel]++;

        for (int candidateIndex = 1; candidateIndex < this.xTrain.size(); candidateIndex++) {
            candidate = this.xTrain.get(candidateIndex);
            dist = distance(querySeries, candidate.firstChannel());
            if (dist < bsfDistance) {
                bsfDistance = dist;
                classCounts = new int[this.numClass];
                classCounts[candidate.classLabel]++;
            } else if (dist == bsfDistance) {
                classCounts[candidate.classLabel]++;
            }
        }

        double bsfCount = -1;
        final int[] out = new int[classCounts.length + 1];
        System.arraycopy(classCounts, 0, out, 1, classCounts.length);
        out[0] = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > bsfCount) {
                bsfCount = classCounts[i];
                out[0] = i;
            }
        }

        return out;
    }

    public PredictionResults loocv0(final Sequences xTrain) throws Exception {
        if (Application.verbose > 0)
            System.out.println("[1-NN] LOOCV0 for " + this.classifierIdentifier);

        this.bestParamId = 0;
        final long start = System.nanoTime();
        PredictionResults loocvResults = loocvAccAndPreds(xTrain, this.bestParamId);
        final long end = System.nanoTime();
        trainingTime = 1.0 * (end - start) / 1e9;

        loocvResults.setTime(start, end);
        loocvResults.setParamId(bestParamId);
        loocvResults.setTrainTest("train");

        if (Application.verbose > 0)
            System.out.printf("[1-NN] LOOCV0 Results: %s, Acc=%.5f, Time=%s%n",
                    getParamInformationString(), loocvResults.accuracy, loocvResults.doTime());

        isTrain = true;

        return loocvResults;
    }

    public PredictionResults loocv0(final double nSample, final Sequences xTrain) throws Exception {

        if (Application.verbose > 0)
            System.out.println("[1-NN] LOOCV0 for " + this.classifierIdentifier);

        this.bestParamId = 0;
        final long start = System.nanoTime();
        PredictionResults loocvResults = loocvAccAndPreds(xTrain, this.bestParamId);
        final long end = System.nanoTime();
        trainingTime = 1.0 * (end - start) / 1e9;

        loocvResults.setTime(start, end);
        loocvResults.setParamId(bestParamId);
        loocvResults.setTrainTest("train");

        if (Application.verbose > 0)
            System.out.printf("[1-NN] LOOCV0 Results: %s, Acc=%.5f, Time=%s%n",
                    getParamInformationString(), loocvResults.accuracy, loocvResults.doTime());

        isTrain = true;

        return loocvResults;
    }

    public PredictionResults loocv(final Sequences xTrain) throws Exception {
        PredictionResults bestResults = new PredictionResults();
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


    public PredictionResults loocv(final double nSample, final Sequences xTrain) throws Exception {
        final Sequences stratified = Sequences.stratifySubset(xTrain, nSample);
        final int[] stratifiedIndex = stratified.indices;

        PredictionResults bestResults = new PredictionResults();
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
            PredictionResults loocvResults = loocvAccAndPreds(xTrain, stratifiedIndex, paramId);
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

    protected PredictionResults loocvAccAndPreds(final Sequences xTrain, final int paramId) throws Exception {
        this.setParamsFromParamId(paramId);

        final int trainSize = xTrain.size();

        trainClassCounts = new int[trainSize][numClass];
        trainPreds = new int[trainSize];
        trainCorrect = 0;

        if (this.nThreads == 1) {
            for (int i = 0; i < trainSize; i++) {
                final Sequence query = xTrain.get(i);
                final int[] a = this.classifyLoocv(query, i);
                System.arraycopy(a, 1, trainClassCounts[i], 0, numClass);
                trainPreds[i] = a[0];
                if (trainPreds[i] == query.classLabel) trainCorrect++;
            }
        } else {
            List<Callable<Integer>> tasks = new ArrayList<>();
            int step = Math.max(trainSize / nThreads, trainSize);
            int start = 0;
            int end = Math.min(trainSize - 1, start + step);
            for (int i = 0; i < this.nThreads; i++) {
                int[] index = new int[end - start + 1];
                for (int j = start; j <= end; j++) index[j - start] = j;
                tasks.add(new OneNNLooCVTask(index, i));
                start = end + 1;
                end = Math.min(trainSize - 1, start + step);
            }
            final MultiThreadedTask parallelTasks = new MultiThreadedTask(this.nThreads);
            MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
            parallelTasks.getExecutor().shutdown();
        }
        final double acc = (double) trainCorrect / trainSize;
        return new PredictionResults(trainCorrect, acc, trainPreds, trainClassCounts);
    }

    protected PredictionResults loocvAccAndPreds(final Sequences xTrain,
                                                 final int[] stratifiedIndex,
                                                 final int paramId) throws Exception {
        this.setParamsFromParamId(paramId);

        final int sampleSize = stratifiedIndex.length;

        trainClassCounts = new int[sampleSize][numClass];
        trainPreds = new int[sampleSize];
        trainCorrect = 0;

        if (this.nThreads == 1) {
            for (int i = 0; i < sampleSize; i++) {
                final Sequence query = xTrain.get(stratifiedIndex[i]);
                final int[] a = this.classifyLoocv(query, stratifiedIndex[i]);
                System.arraycopy(a, 1, trainClassCounts[i], 0, numClass);
                trainPreds[i] = a[0];
                if (trainPreds[i] == query.classLabel) trainCorrect++;
            }
        } else {
            List<Callable<Integer>> tasks = new ArrayList<>();
            int step = Math.max(sampleSize / nThreads, sampleSize);
            int start = 0;
            int end = Math.min(sampleSize - 1, start + step);
            for (int i = 0; i < this.nThreads; i++) {
                int[] index = new int[end - start + 1];
                int[] sIndex = new int[end - start + 1];
                if (start > end) continue;
                for (int j = start; j <= end; j++) {
                    index[j - start] = j;
                    sIndex[j - start] = stratifiedIndex[j];
                }
                tasks.add(new OneNNLooCVTask(index, sIndex, i));
                start = end + 1;
                end = Math.min(sampleSize - 1, start + step);
            }
            final MultiThreadedTask parallelTasks = new MultiThreadedTask(this.nThreads);
            MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
            parallelTasks.getExecutor().shutdown();
        }
        final double acc = (double) trainCorrect / sampleSize;
        return new PredictionResults(trainCorrect, acc, trainPreds, trainClassCounts);
    }


    public int[] classifyLoocv(final Sequence query, final int queryIndex) throws Exception {
        int[] classCounts = new int[this.numClass];

        double dist;

        int candidateIndex = (queryIndex > 0) ? 0 : 1;
        int nextIndex = candidateIndex + 1;
        double[] querySeries = query.firstChannel();
        Sequence candidate = this.xTrain.get(candidateIndex);
        double bsfDistance = distance(querySeries, candidate.firstChannel());
        classCounts[candidate.classLabel]++;

        for (candidateIndex = nextIndex; candidateIndex < xTrain.size(); candidateIndex++) {
            if (queryIndex == candidateIndex)
                continue;
            candidate = xTrain.get(candidateIndex);
            dist = distance(querySeries, candidate.firstChannel(), bsfDistance);
            if (dist < bsfDistance) {
                bsfDistance = dist;
                classCounts = new int[this.numClass];
                classCounts[candidate.classLabel]++;
            } else if (dist == bsfDistance) {
                classCounts[candidate.classLabel]++;
            }
        }

        double bsfCount = -1;
        final int[] out = new int[classCounts.length + 1];
        System.arraycopy(classCounts, 0, out, 1, classCounts.length);
        out[0] = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > bsfCount) {
                bsfCount = classCounts[i];
                out[0] = i;
            }
        }
        return out;
    }

    @Override
    public PredictionResults evaluate(final Sequences xTest) throws Exception {
        this.setTestData(xTest);

        for (int i = 0; i < this.useDerivative; i++)
            this.xTest = DerivativeFilter.getFirstDerivative(this.xTest);

        this.setParamsFromParamId(bestParamId);
        final int testSize = xTest.size();
        testClassCounts = new int[testSize][numClass];
        testPreds = new int[testSize];
        testCorrect = 0;

        if (this.nThreads == 1) {
            testResults = super.evaluate(this.xTest);
        } else {
            final long startTime = System.nanoTime();

            List<Callable<Integer>> tasks = new ArrayList<>();
            int step = testSize / nThreads;
            int start = 0;
            int end = Math.min(testSize - 1, start + step);
            for (int i = 0; i < this.nThreads; i++) {
                int[] index = new int[end - start + 1];
                for (int j = start; j <= end; j++) index[j - start] = j;
                tasks.add(new OneNNEvaluateTask(index, i));
                start = end + 1;
                end = Math.min(testSize - 1, start + step);
            }
            final MultiThreadedTask parallelTasks = new MultiThreadedTask(this.nThreads);
            try {
                MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
            } catch (Exception e) {
                e.printStackTrace();
            }
            parallelTasks.getExecutor().shutdown();
            final long endTime = System.nanoTime();

            final double acc = 1.0 * testCorrect / testSize;
            testResults = new PredictionResults(testCorrect, acc, testPreds, testClassCounts, "test");
            testResults.setTime(startTime, endTime);
        }
        testResults.setParamId(bestParamId);

        return testResults;
    }


    public void setParamsFromParamId(final int paramId) {
        this.paramId = paramId;
    }

    public void setRandomParams(Random rand) {
        this.setParamsFromParamId(rand.nextInt(nParams));
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] nThread: " + nThreads +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    public static OneNearestNeighbour init(HashMap<Transforms.TimeSeriesTransforms, Sequences> xTrain, String classifierName) {
        OneNearestNeighbour classifier = init(-1, classifierName);
        return classifier;
    }

    public static OneNearestNeighbour init(int paramId, String classifierName) {
        OneNearestNeighbour classifier;
        switch (classifierName) {
            // =============== (13) ED
            case "d3-ED1NN":
                classifier = new ED1NN(paramId, 3);
                break;
            case "d2-ED1NN":
                classifier = new ED1NN(paramId, 2);
                break;
            case "d1-ED1NN":
                classifier = new ED1NN(paramId, 1);
                break;
            case "ED1NN":
                classifier = new ED1NN(paramId);
                break;

            // =============== (10) Minkowski
            case "d3-Minkowski1NN-0.5":
                classifier = new Minkowski1NN(0.5, 3);
                break;
            case "d2-Minkowski1NN-0.5":
                classifier = new Minkowski1NN(0.5, 2);
                break;
            case "d1-Minkowski1NN-0.5":
                classifier = new Minkowski1NN(0.5, 1);
                break;
            case "Minkowski1NN-0.5":
                classifier = new Minkowski1NN(0.5);
                break;

            case "d3-Minkowski1NN-1.0":
                classifier = new Minkowski1NN(1.0, 3);
                break;
            case "d2-Minkowski1NN-1.0":
                classifier = new Minkowski1NN(1.0, 2);
                break;
            case "d1-Minkowski1NN-1.0":
                classifier = new Minkowski1NN(1.0, 1);
                break;
            case "Minkowski1NN-1.0":
                classifier = new Minkowski1NN(1.0);
                break;

            case "d3-Minkowski1NN":
                classifier = new Minkowski1NN(paramId, 3);
                break;
            case "d2-Minkowski1NN":
                classifier = new Minkowski1NN(paramId, 2);
                break;
            case "d1-Minkowski1NN":
                classifier = new Minkowski1NN(paramId, 1);
                break;
            case "Minkowski1NN":
                classifier = new Minkowski1NN(paramId);
                break;
            // =============== (9) LCSS
            case "d3-LCSS1NN":
                classifier = new LCSS1NN(paramId, 3);
                break;
            case "d2-LCSS1NN":
                classifier = new LCSS1NN(paramId, 2);
                break;
            case "d1-LCSS1NN":
                classifier = new LCSS1NN(paramId, 1);
                break;
            case "LCSS1NN":
                classifier = new LCSS1NN(paramId);
                break;

            // =============== (5) ADTW+
            case "d3-ADTW+1NN":
                classifier = new ADTWPlus1NN(paramId, -1, 3);
                break;
            case "d2-ADTW+1NN":
                classifier = new ADTWPlus1NN(paramId, -1, 2);
                break;
            case "d1-ADTW+1NN":
                classifier = new ADTWPlus1NN(paramId, -1, 1);
                break;
            case "ADTW+1NN":
                classifier = new ADTWPlus1NN(paramId, -1);
                break;

            case "d3-ADTW1NN-0.5":
                classifier = new ADTWPlus1NN(paramId, 0.5, 3);
                break;
            case "d2-ADTW1NN-0.5":
                classifier = new ADTWPlus1NN(paramId, 0.5, 2);
                break;
            case "d1-ADTW1NN-0.5":
                classifier = new ADTWPlus1NN(paramId, 0.5, 1);
                break;
            case "ADTW1NN-0.5":
                classifier = new ADTWPlus1NN(paramId, 0.5);
                break;

            case "d3-ADTW1NN-1.0":
                classifier = new ADTWPlus1NN(paramId, 1, 3);
                break;
            case "d2-ADTW1NN-1.0":
                classifier = new ADTWPlus1NN(paramId, 1, 2);
                break;
            case "d1-ADTW1NN-1.0":
                classifier = new ADTWPlus1NN(paramId, 1, 1);
                break;
            case "ADTW1NN-1.0":
                classifier = new ADTWPlus1NN(paramId, 1);
                break;


            // =============== (4) ADTW
            case "d3-ADTW1NN":
                classifier = new ADTW1NN(paramId, 3);
                break;
            case "d2-ADTW1NN":
                classifier = new ADTW1NN(paramId, 2);
                break;
            case "d1-ADTW1NN":
                classifier = new ADTW1NN(paramId, 1);
                break;
            case "ADTW1NN":
                classifier = new ADTW1NN(paramId);
                break;

            // =============== (2) CDTW+
            case "d3-CDTW+1NN":
                classifier = new CDTWPlus1NN(paramId, -1, 3);
                break;
            case "d2-CDTW+1NN":
                classifier = new CDTWPlus1NN(paramId, -1, 2);
                break;
            case "d1-CDTW+1NN":
                classifier = new CDTWPlus1NN(paramId, -1, 1);
                break;
            case "CDTW+1NN":
                classifier = new CDTWPlus1NN(paramId, -1);
                break;


            case "d3-CDTW1NN-0.5":
                classifier = new CDTWPlus1NN(paramId, 0.5, 3);
                break;
            case "d2-CDTW1NN-0.5":
                classifier = new CDTWPlus1NN(paramId, 0.5, 2);
                break;
            case "d1-CDTW1NN-0.5":
                classifier = new CDTWPlus1NN(paramId, 0.5, 1);
                break;
            case "CDTW1NN-0.5":
                classifier = new CDTWPlus1NN(paramId, 0.5);
                break;

            case "d3-CDTW1NN-1.0":
                classifier = new CDTWPlus1NN(paramId, 1, 3);
                break;
            case "d2-CDTW1NN-1.0":
                classifier = new CDTWPlus1NN(paramId, 1, 2);
                break;
            case "d1-CDTW1NN-1.0":
                classifier = new CDTWPlus1NN(paramId, 1, 1);
                break;
            case "CDTW1NN-1.0":
                classifier = new CDTWPlus1NN(paramId, 1);
                break;

            // =============== (1) CDTW
            case "d3-CDTW1NN":
                classifier = new CDTW1NN(paramId, 3);
                break;
            case "d2-CDTW1NN":
                classifier = new CDTW1NN(paramId, 2);
                break;
            case "d1-CDTW1NN":
                classifier = new CDTW1NN(paramId, 1);
                break;
            default:
                // CDTW-1NN
                classifier = new CDTW1NN(paramId);
                break;
        }

        return classifier;
    }
}
