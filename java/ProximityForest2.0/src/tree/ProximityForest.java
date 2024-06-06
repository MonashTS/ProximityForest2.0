package tree;

import application.Application;
import datasets.Sequence;
import datasets.Sequences;
import hydra.Hydra;
import nearestNeighbour.ADTW1NN;
import results.PredictionResults;
import transforms.DerivativeFilter;
import tree.results.PFResults;
import tree.splitters.NodeSplitter;
import utils.MultiThreadedTask;
import utils.Tools;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

import static transforms.Transforms.TimeSeriesTransforms.*;
import static tree.splitters.NodeSplitter.SplitterType.PF;
import static tree.splitters.NodeSplitter.SplitterType.PF2;
import static tree.splitters.PF2Splitter.enabledDistancesCostFunction;

public class ProximityForest extends TimeSeriesClassifier {
    /**
     * Parallel training task
     * Train each tree in parallel
     */
    protected static class PFTrainingTask implements Callable<Integer> {
        protected Tree tree;
        protected int treeId; // thread id

        public PFTrainingTask(final Tree tree, final int treeId) {
            this.tree = tree;
            this.treeId = treeId;
        }

        @Override
        public Integer call() throws Exception {
            if (Application.verbose > 1) System.out.println("[Tree " + treeId + "] started fit");
            this.tree.fit(this.tree.trainIndices);

            if (Application.verbose > 1) System.out.println("[Tree " + treeId + "] fit completed");
            else System.out.print(treeId + ",");

            return null;
        }
    }

    /**
     * Parallel prediction task
     * Do prediction on query from each tree in parallel
     */
    protected static class PFPredictionTask implements Callable<int[]> {
        protected Tree tree;
        protected int treeId; // thread id
        protected Sequence query;

        public PFPredictionTask(final Tree tree, final Sequence query, final int treeId) {
            this.tree = tree;
            this.treeId = treeId;
            this.query = query;
        }

        @Override
        public int[] call() {
            return this.tree.predict(query);
        }
    }

    /**
     * Parallel evaluation task
     * Do evaluation on test dataset from each tree in parallel
     */
    protected static class PFEvaluateTask implements Callable<Integer> {
        protected Tree tree;
        protected int treeId;
        protected Sequences dataset;

        public PFEvaluateTask(final Tree tree, final Sequences dataset, final int treeId) {
            this.tree = tree;
            this.treeId = treeId;
            this.dataset = dataset;
        }

        @Override
        public Integer call() {
            this.tree.evaluate(this.dataset);
            System.out.print(treeId + ",");
            return null;
        }
    }

    protected Tree[] trees;
    public EnsembleVotingScheme ensembleVotingScheme = EnsembleVotingScheme.majority; // can be set
    public int maxDepth = -1;
    public int leafGiniThreshold = 0;
    public int minNodeSize = 1;
    public int numTrees;
    public String splitterConfig;
    public HashMap<NodeSplitter.SplitterType, Integer> splittersPerNode;
    public HashMap<NodeSplitter.SplitterType, Boolean> splittersEnabled;

    // distance - PF
    public static int adtwNParams = 100;
    public static int adtwNSamples = ADTW1NN.nSamples;
    public static int adtwExponents = ADTW1NN.exponent;

    // dictionary - HYDRA
    public Hydra hydraTransformer; // to calculate hydra features
    public double[][][] hydraFeatures; // hydra features by groups

    public final String name = "ProximityForest";

    // results
    public PFResults trainResults;
    public PFResults testResults;

    // constructors
    public ProximityForest(final int numTrees, final HashMap<NodeSplitter.SplitterType, Integer> splittersPerNode) {
        this.numTrees = numTrees;
        this.splitterConfig = "n=" + numTrees;
        this.splittersPerNode = splittersPerNode;
        this.splittersEnabled = new HashMap<>();
        for (NodeSplitter.SplitterType type : splittersPerNode.keySet()) {
            if (splittersPerNode.get(type) <= 0) this.splittersEnabled.put(type, false);
            else {
                this.splittersEnabled.put(type, true);
                this.splitterConfig = String.format("%s:%s=%d", this.splitterConfig, type, this.splittersPerNode.get(type));
            }
        }
        this.classifierIdentifier = name;
        this.rand = Application.rand;
    }

    private void doHydraTransform(final Sequences xTrain) {
        // do Hydra transform
        final long startTime = System.nanoTime();
        this.hydraTransformer = new Hydra();
        this.hydraTransformer.fit(xTrain);
        final double[][] transforms = this.hydraTransformer.transform(xTrain);
        final Sequences hydraFeatures = new Sequences();
        for (int i = 0; i < transforms.length; i++) {
            final Sequence s = new Sequence(transforms[i], xTrain.get(i).classLabel);
            s.type = hydra;
            hydraFeatures.add(s, i);
        }
        trainResults.setHydraTransformTime(startTime, System.nanoTime());

        this.trainDataset.put(hydra, hydraFeatures);
        if (Application.verbose > 0)
            System.out.println("[ProximityForest] HYDRA Transform time: " + Tools.doTime(this.trainResults.hydraTransformTime));
    }

    @Override
    public PredictionResults fit(final Sequences xTrain) throws Exception {
        // always make sure that class labels in xTrain are from 0 to C
        this.setTrainingData(xTrain);

        long startTime;

        // initialise
        this.trainResults = new PFResults();
        this.trainResults.setTrainTest("train");

        this.trees = new Tree[this.numTrees];

        // distance-based
        xTrain.initLCSSParam();
        if (this.splittersEnabled.get(PF)) {
            this.trainDataset.put(d1, DerivativeFilter.getFirstDerivative(xTrain));
        } else if (this.splittersEnabled.get(PF2)) {
            for (double ge : enabledDistancesCostFunction)
                xTrain.initADTWWeights(adtwNSamples, adtwNParams, adtwExponents, ge);

            final Sequences x = DerivativeFilter.getFirstDerivative(xTrain);
            x.initLCSSParam();
            for (double ge : enabledDistancesCostFunction)
                x.initADTWWeights(adtwNSamples, adtwNParams, adtwExponents, ge);
            this.trainDataset.put(d1, x);

            doHydraTransform(xTrain);
            final Sequences transforms = this.trainDataset.get(hydra);
            for (int i = 0; i < transforms.size(); i++) {
                xTrain.get(i).transforms = new HashMap<>();
                xTrain.get(i).transforms.put(hydra, transforms.get(i).firstChannel());
            }
        }

        this.trainDataset.put(raw, xTrain);

        if (this.nThreads == 1) {
            startTime = System.nanoTime();
            for (int i = 0; i < this.numTrees; i++) {
                this.trees[i] = new Tree(i, this);
                this.trees[i].fit(this.trainIndices);
            }
            this.trainResults.setTime(startTime, System.nanoTime());
        } else {
            startTime = System.nanoTime();
            final List<Callable<Integer>> tasks = new ArrayList<>();
            for (int i = 0; i < this.numTrees; i++) {
                this.trees[i] = new Tree(i, this);
                tasks.add(new PFTrainingTask(this.trees[i], i));
            }
            final MultiThreadedTask parallelTasks = new MultiThreadedTask(this.nThreads);
            MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
            this.trainResults.setTime(startTime, System.nanoTime());
            System.out.println();
            parallelTasks.getExecutor().shutdown();
        }
        this.isTrain = true;

        HashMap<String, Integer> splitterCounts = new HashMap<>();
        for (Tree tree : this.trees) {
            for (String key : tree.distanceCount.keySet()) {
                if (!splitterCounts.containsKey(key)) splitterCounts.put(key, 0);

                splitterCounts.put(key, splitterCounts.get(key) + tree.distanceCount.get(key));
            }
        }
        for (String key : splitterCounts.keySet())
            System.out.println("[" + key + "]: " + splitterCounts.get(key));

        for (int i = 0; i < this.trees.length; i++) {
            this.trainResults.leafCount += trees[i].leafCount;

            for (NodeSplitter.SplitterType key : trees[i].splitterCount.keySet()) {
                if (!this.trainResults.splitterCount.containsKey(key)) {
                    this.trainResults.splitterCount.put(key, trees[i].splitterCount.get(key));
                    this.trainResults.splitterTime.put(key, trees[i].splitterTime.get(key));
                } else {
                    this.trainResults.splitterCount.put(key, trees[i].splitterCount.get(key) + 1);
                    this.trainResults.splitterTime.put(key, trees[i].splitterTime.get(key) + 1);
                }
            }

            this.trainResults.totalTrainTime += trees[i].trainResults.elapsedTimeNanoSeconds;
            if (Application.verbose > 1) {
                final StringBuilder s = new StringBuilder("Tree " + i + " -- " + Tools.doTime(trees[i].trainResults.elapsedTimeNanoSeconds) + ":NLEAFS-" + trees[i].leafCount);
                for (NodeSplitter.SplitterType st : trees[i].splitterCount.keySet())
                    s.append(st).append("-").append(trees[i].splitterTime.get(st)).append("(").append(trees[i].splitterCount.get(st)).append(")");
                System.out.println(s);
            }
        }
        if (Application.verbose > 0) {
            final StringBuilder s = new StringBuilder("[" + name + "] -- " + this.nThreads + " threads Time: " + Tools.doTime(this.trainResults.elapsedTimeNanoSeconds) + ", Serial Time: " + Tools.doTime(this.trainResults.totalTrainTime) + ":NLEAFS-" + this.trainResults.leafCount);
            for (NodeSplitter.SplitterType st : this.trainResults.splitterCount.keySet()) {
                s.append(":::");
                s.append(st).append("-").append(Tools.doTime(this.trainResults.splitterTime.get(st)));
                s.append("(").append(this.trainResults.splitterCount.get(st)).append(")");
            }
            System.out.println(s);
        }
        return this.trainResults;
    }

    @Override
    public PredictionResults fit(final Sequences xTrain, final double nSample) throws Exception {
        return this.fit(xTrain);
    }

    @Override
    public PFResults evaluate(final Sequences xTest) throws Exception {
        final int testSize = xTest.size();
        final int numClass = this.numClass;

        this.testResults = new PFResults();
        this.testResults.setTrainTest("test");

        this.testClassCounts = new int[testSize][numClass];
        this.testPreds = new int[testSize];
        this.testCorrect = 0;

        this.testDataset = new HashMap<>();
        this.testDataset.put(raw, xTest);

        if (this.splittersEnabled.get(PF2)) {
            final long startTime = System.nanoTime();
            final double[][] transforms = this.hydraTransformer.transform(xTest);
            final Sequences features = new Sequences();
            for (int i = 0; i < transforms.length; i++) {
                final Sequence s = new Sequence(transforms[i], xTest.get(i).classLabel);
                s.type = hydra;
                features.add(s, i);
                if (xTest.get(i).transforms == null) xTest.get(i).transforms = new HashMap<>();
                xTest.get(i).transforms.put(hydra, transforms[i]);
            }

            this.testDataset.put(hydra, features);
            this.testResults.setHydraTransformTime(startTime, System.nanoTime());
        }

        // evaluate test dataset in parallel for each tree
        final long startTime = System.nanoTime();
        final List<Callable<Integer>> tasks = new ArrayList<>();
        for (int i = 0; i < this.numTrees; i++) {
            this.trees[i].testDataset = this.testDataset;
            tasks.add(new PFEvaluateTask(this.trees[i], xTest, i));
        }

        final MultiThreadedTask parallelTasks = new MultiThreadedTask(this.nThreads);
        MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
        final long stopTime = System.nanoTime();
        parallelTasks.getExecutor().shutdown();
        System.out.println();

        // collect the results
        long elapsed = 0;
        for (int i = 0; i < testSize; i++) {
            final ArrayList<int[]> predictionResults = new ArrayList<>();
            for (Tree tree : this.trees) {
                final int[] out = new int[numClass + 1];
                out[0] = tree.testResults.predictions[i];
                System.arraycopy(tree.testResults.classCounts[i], 0, out, 1, numClass);
                predictionResults.add(out);
                if (i == 0) {
                    elapsed += tree.testResults.elapsedTimeNanoSeconds;
                    for (NodeSplitter.SplitterType key : tree.testResults.splitterTime.keySet()) {
                        final long a = tree.testResults.splitterTime.get(key);
                        if (!testResults.splitterTime.containsKey(key)) testResults.splitterTime.put(key, a);
                        else testResults.splitterTime.put(key, testResults.splitterTime.get(key) + a);
                    }
                }
            }
            final int[] a = majorityVoteEnsemble(predictionResults);
            final int predictClass = a[0];
            System.arraycopy(a, 1, this.testClassCounts[i], 0, numClass);
            if (predictClass == xTest.get(i).classLabel) this.testCorrect++;
            this.testPreds[i] = predictClass;
        }

        testResults.nCorrect = testCorrect;
        testResults.accuracy = 1.0 * testCorrect / testSize;
        testResults.predictions = testPreds;
        testResults.classCounts = testClassCounts;
        testResults.setTime(startTime, stopTime);
        testResults.totalTestTime = elapsed;

        if (Application.verbose > 0) {
            final StringBuilder s = new StringBuilder("[" + name + "] -- " + this.nThreads + " threads Time: " + Tools.doTime(this.testResults.elapsedTimeNanoSeconds) + ", Serial Time: " + Tools.doTime(this.testResults.totalTrainTime) + ":NLEAFS-" + this.testResults.leafCount);
            for (NodeSplitter.SplitterType st : this.testResults.splitterTime.keySet()) {
                s.append(":::");
                s.append(st).append("-").append(Tools.doTime(this.testResults.splitterTime.get(st)));
            }
            System.out.println(s);
        }
        return testResults;
    }


    /**
     * Predict the class of the given query
     *
     * @param query - query time series
     * @return class labels array - first element is the predicted class.
     * @throws Exception
     */
    @Override
    public int[] predict(Sequence query) throws Exception {
        // given a query, predict the class
        final List<Callable<int[]>> tasks = new ArrayList<>();
        for (int i = 0; i < this.numTrees; i++)
            tasks.add(new PFPredictionTask(this.trees[i], query, i));

        final MultiThreadedTask parallelTasks = new MultiThreadedTask(this.nThreads);
        final ArrayList<int[]> predictionResults = MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
        parallelTasks.getExecutor().shutdown();

        return majorityVoteEnsemble(predictionResults);
    }


    public void setTrainingData(final Sequences xTrain) {
        this.trainDataset = new HashMap<>();
        this.numClass = xTrain.getNumClasses();
        this.trainIndices = Tools.getDatasetIndices(xTrain);
        this.yTrain = xTrain.getLabels();
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier + "\n[CLASSIFIER SUMMARY] nThread: " + nThreads + "\n[CLASSIFIER SUMMARY] nTrees: " + numTrees + "\n[CLASSIFIER SUMMARY] splitterConfig: " + splitterConfig;
    }
}
