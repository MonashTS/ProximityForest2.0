package tree;

import application.Application;
import results.PredictionResults;
import utils.Tools;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.Callable;

/**
 * An abstract class for any ML classifier
 */
public abstract class AbstractClassifier {

    /**
     * A task for evaluating a classifier
     */
    protected static class EvaluateTask implements Callable<Integer> {
        protected int[] index; // index to evaluate from test data
        protected int threadId; // thread id

        public EvaluateTask(int[] index, int threadId) {
            this.index = index;
            this.threadId = threadId;
        }

        @Override
        public Integer call() throws Exception {
            System.out.println("[Thread_" + threadId + "] Complete");
            return null;
        }
    }

    public enum EnsembleVotingScheme {
        majority,
        prob
    }

    public double[][] xTrain;
    public int[] yTrain;
    public int[] trainIndices;
    public int numClass;
    public int[][] trainClassCounts;
    public int[] trainPreds;
    public int trainCorrect;
    public int[][] testClassCounts;
    public int[] testPreds;
    public int testCorrect;
    public String classifierIdentifier;
    public double trainingTime;
    public int nThreads = 1;
    public boolean isTrain = false;
    protected HashMap<Integer, Integer> trainClassDistribution;
    protected HashMap<Integer, ArrayList<Integer>> trainClassIndex;

    public PredictionResults trainResults;
    public PredictionResults testResults;

    public Random rand;

    public void summary() {
        System.out.println(this);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] nThread: " + nThreads;
    }
    public void setClassDistribution() {
        ArrayList<Integer> temp;
        this.trainClassDistribution = new HashMap<>();
        this.trainClassIndex = new HashMap<>();
        for (int i = 0; i < this.xTrain.length; i++) {
            final int label = this.yTrain[i];

            if (trainClassDistribution.containsKey(label)) {
                trainClassDistribution.put(label, trainClassDistribution.get(label) + 1);
                temp = trainClassIndex.get(label);
            } else {
                trainClassDistribution.put(label, 1);
                temp = new ArrayList<>();
            }
            temp.add(i);
            trainClassIndex.put(label, temp);
        }
    }

    public int getNumClasses() {
        if (this.trainClassDistribution == null)
            setClassDistribution();

        return this.trainClassDistribution.size();
    }

    // --- --- --- Set functions
    public void setTrainingData(final double[][] xTrain, final int[] yTrain) {
        this.xTrain = xTrain;
        this.yTrain = yTrain;

        this.numClass = this.getNumClasses();
    }

    public void setThreads(int threads) {
        // make sure n_threads is less than n_cpu
        if (threads < 0) threads = Runtime.getRuntime().availableProcessors();
        else threads = Math.min(threads, Runtime.getRuntime().availableProcessors());
        threads = Math.max(threads, 1);

        if (Application.verbose > 0)
            System.out.println("Setting nThreads to " + threads);
        this.nThreads = threads;
    }

    public void setVerbose(int verbose) {
        Application.verbose = verbose;
    }

    public int[] majorityVoteEnsemble(final ArrayList<int[]> forestResults) {
        final int[] classCounts = new int[this.numClass];

        for (int[] treeClassPreds : forestResults)
            for (int i = 0; i < classCounts.length; i++)
                classCounts[i] += treeClassPreds[i + 1];

        return Tools.generateOutputs(classCounts);
    }
}
