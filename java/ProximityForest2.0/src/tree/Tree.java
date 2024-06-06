package tree;

import application.Application;
import datasets.Sequence;
import datasets.Sequences;
import hydra.Hydra;
import nearestNeighbour.ADTW1NN;
import results.PredictionResults;
import transforms.DerivativeFilter;
import transforms.Transforms;
import tree.results.PFResults;
import tree.splitters.NodeSplitter.SplitterType;
import utils.Tools;

import java.util.HashMap;
import java.util.List;

import static transforms.Transforms.TimeSeriesTransforms.*;
import static tree.splitters.NodeSplitter.SplitterType.PF;
import static tree.splitters.NodeSplitter.SplitterType.PF2;
import static tree.splitters.PF2Splitter.enabledDistancesCostFunction;

public class Tree extends TimeSeriesClassifier {
    public int treeId;

    // distance - PF
    public static int adtwNParams = 100;
    public static int adtwNSamples = ADTW1NN.nSamples;
    public static int adtwExponents = ADTW1NN.exponent;

    protected int numNodes = 0;
    protected int numLeaves = 0;
    protected int depth = 0;
    public int leafCount = 0;

    protected transient ProximityForest forest;
    private Node root;
    public EnsembleVotingScheme ensembleVotingScheme = EnsembleVotingScheme.majority;
    public int maxDepth = -1;
    public int leafGiniThreshold = 0;
    public int minNodeSize = 1;

    public HashMap<SplitterType, Integer> splittersPerNode;
    public HashMap<SplitterType, Boolean> splittersEnabled;
    public transient HashMap<String, Integer> distanceCount = new HashMap<>();
    public transient HashMap<SplitterType, Integer> splitterCount = new HashMap<>();
    public transient HashMap<SplitterType, Long> splitterTime = new HashMap<>();


    // dictionary - HYDRA
    public Hydra hydraTransformer;
    public double[][][] hydraFeatures;
    public HashMap<Sequence, double[][]> cachedHydraFeatures;
    public HashMap<Transforms.TimeSeriesTransforms, List<int[]>> quantIntervals;

    // interval - RISE
    public int minRiseInterval = 16;
    public int riseNumIntervals = 0; //if 0 its determined auto,
    // interval - Quant
    public int quantDepth = 6;
    public int quantDiv = 4;
    public String name = "PFTree";
    protected String splitterConfig;

    // results
    public PFResults trainResults;
    public PFResults testResults;

    // constructors
    public Tree(final HashMap<SplitterType, Integer> splittersPerNode) {
        this.splitterConfig = "";
        this.splittersPerNode = splittersPerNode;
        this.splittersEnabled = new HashMap<>();
        for (SplitterType type : splittersPerNode.keySet()) {
            if (splittersPerNode.get(type) <= 0) this.splittersEnabled.put(type, false);
            else {
                this.splittersEnabled.put(type, true);
                this.splitterConfig = String.format("%s:%s=%d", this.splitterConfig, type, this.splittersPerNode.get(type));
            }
        }
        this.classifierIdentifier = name;
        this.rand = Application.rand;

        this.resetTime();
    }


    public Tree(final int treeId, final ProximityForest forest) {
        this(forest.splittersPerNode);
        this.forest = forest;
        this.treeId = treeId;

        this.rand = forest.rand;
        this.leafGiniThreshold = forest.leafGiniThreshold;
        this.maxDepth = forest.maxDepth;
        this.minNodeSize = forest.minNodeSize;
        this.hydraTransformer = forest.hydraTransformer;
        this.hydraFeatures = forest.hydraFeatures;
        this.ensembleVotingScheme = forest.ensembleVotingScheme;
        this.trainDataset = forest.trainDataset;
        this.testDataset = forest.testDataset;
        this.trainIndices = forest.trainIndices;
        this.yTrain = forest.yTrain;
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
            System.out.println("[TREE] HYDRA Transform time: " + Tools.doTime(this.trainResults.hydraTransformTime));
    }

    @Override
    public PredictionResults fit(final Sequences xTrain) throws Exception {
        this.setTrainingData(xTrain);

        this.resetTime();

        this.trainResults = new PFResults();
        this.trainResults.setTrainTest("train");

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

        final long startTime = System.nanoTime();
        this.fit(this.trainIndices);
        this.trainResults.setTime(startTime, System.nanoTime());

        if (Application.verbose > 0) {
            final StringBuilder s = new StringBuilder("[" + name + "] -- " + this.nThreads + " threads Time: " + Tools.doTime(this.trainResults.elapsedTimeNanoSeconds) + ", Serial Time: " + Tools.doTime(this.trainResults.totalTrainTime) + ":NLEAFS-" + this.trainResults.leafCount);
            for (SplitterType st : this.trainResults.splitterCount.keySet()) {
                s.append(":::");
                s.append(st).append("-").append(Tools.doTime(this.trainResults.splitterTime.get(st)));
                s.append("(").append(this.trainResults.splitterCount.get(st)).append(")");
            }
            System.out.println(s);
        }
        return trainResults;
    }

    @Override
    public PredictionResults fit(final Sequences xTrain, final double nSample) throws Exception {
        return this.fit(xTrain);
    }

    public PredictionResults fit(final int[] indices) throws Exception {
        this.resetTime();

        trainResults = new PFResults();
        trainResults.setTrainTest("train");

        final long start = System.nanoTime();
        if (this.treeId == -1) this.root = new Node(this, null, -1);
        else this.root = new Node(this, null, numNodes++);

        this.root.fit(indices);

        final long end = System.nanoTime();
        trainingTime = 1.0 * (end - start) / 1e9;
        trainResults.setTime(start, end);

        isTrain = true;
        this.trainResults.leafCount = this.leafCount;
        this.trainResults.totalTrainTime = trainResults.elapsedTimeNanoSeconds;
        this.trainResults.splitterCount = this.splitterCount;
        this.trainResults.splitterTime = this.splitterTime;

        return trainResults;
    }


    @Override
    public PredictionResults evaluate(final Sequences xTest) {
        this.resetTime();

        final int testSize = xTest.size();
        final int numClass = this.getNumClass();

        this.testResults = new PFResults();
        this.testResults.setTrainTest("test");

        this.testClassCounts = new int[testSize][numClass];
        this.testPreds = new int[testSize];
        this.testCorrect = 0;

        if (this.testDataset == null) {
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
        }

        final long startTime = System.nanoTime();
        for (int i = 0; i < testSize; i++) {
            final int[] a = predictSeries(xTest.get(i), i);
            final int predictClass = a[0];
            System.arraycopy(a, 1, testClassCounts[i], 0, numClass);
            if (predictClass == xTest.get(i).classLabel) testCorrect++;
            testPreds[i] = predictClass;
        }
        final long stopTime = System.nanoTime();

        testResults.nCorrect = testCorrect;
        testResults.accuracy = 1.0 * testCorrect / testSize;
        testResults.predictions = testPreds;
        testResults.classCounts = testClassCounts;
        testResults.setTime(startTime, stopTime);
        testResults.splitterTime = splitterTime;

        return testResults;
    }

    @Override
    public int[] predict(final Sequence query) {
        this.resetTime();
        return predictSeries(query);
    }

    protected int[] predictSeries(final Sequence query) {
        Node node = this.root;
        final int[] classCounts = new int[this.forest.numClass];
        int childNodeIndex = 0;

        while (node != null && !node.isLeaf) {
            childNodeIndex = node.predict(query);

            node = node.getChildren(childNodeIndex);
            if (node == null) {
                System.out.println("null node found: " + childNodeIndex);
                classCounts[childNodeIndex] = 1;
                return Tools.generateOutputs(classCounts);
            }
        }

        // null node found, returning exemplar label
        if (node == null) {
            classCounts[childNodeIndex] = 1;
            return Tools.generateOutputs(classCounts);
        } else if (node.getLabel() == null) {
            classCounts[childNodeIndex] = 1;
            return Tools.generateOutputs(classCounts);
        }
        classCounts[node.getLabel()] = 1;

        return Tools.generateOutputs(classCounts);
    }

    protected int[] predictSeries(final Sequence query, int queryIndex) {
        Node node = this.root;
        final int[] classCounts = new int[this.forest.numClass];
        int childNodeIndex = 0;

        while (node != null && !node.isLeaf) {
            if (node.bestSplitter.name.equals("HYDRA"))
                childNodeIndex = node.predict(this.getTestSeries(queryIndex, hydra));
            else childNodeIndex = node.predict(query);

            node = node.getChildren(childNodeIndex);
            if (node == null) {
                // note that this does not apply to extra tree or random tree splitters.
                // each splitters should return node index wrt the classes
                // i.e. each children key is the class index
                System.out.println("null node found: " + childNodeIndex);
                classCounts[childNodeIndex] = 1;
                return Tools.generateOutputs(classCounts);
            }
        }

        // null node found, returning exemplar label
        if (node == null) {
            classCounts[childNodeIndex] = 1;
            return Tools.generateOutputs(classCounts);
        } else if (node.getLabel() == null) {
            classCounts[childNodeIndex] = 1;
            return Tools.generateOutputs(classCounts);
        }
        classCounts[node.getLabel()] = 1;

        return Tools.generateOutputs(classCounts);
    }

    protected void resetTime() {
        for (SplitterType s : splittersEnabled.keySet()) {
            if (splittersEnabled.get(s)) {
                this.splitterCount.put(s, 0);
                this.splitterTime.put(s, 0L);
            }
        }
    }

    public int getNumClass() {
        if (forest != null) return forest.numClass;
        return this.numClass;
    }


    public HashMap<Transforms.TimeSeriesTransforms, Sequences> getTrainTS() {
        return this.trainDataset;
    }

    public int[] getTrainLabels() {
        return this.trainDataset.get(raw).getLabels();
    }

    public Sequence getTrainSeries(final int i, final Transforms.TimeSeriesTransforms transform) {
        return trainDataset.get(transform).get(i);
    }

    public Sequence getTestSeries(final int i, final Transforms.TimeSeriesTransforms transform) {
        return testDataset.get(transform).get(i);
    }

    public int getTrainLabel(final int i) {
        return this.trainDataset.get(raw).get(i).classLabel;
    }

    public List<int[]> getMetaIntervals(Transforms.TimeSeriesTransforms repr) {
        return quantIntervals.get(repr);
    }

    public void setTrainingData(final Sequences xTrain) {
        this.trainDataset = new HashMap<>();
        this.numClass = xTrain.getNumClasses();
        this.trainIndices = Tools.getDatasetIndices(xTrain);
        this.yTrain = xTrain.getLabels();
    }

    public boolean isPfEnabled() {
        return this.splittersEnabled.get(PF);
    }

    public boolean isPf2HEnabled() {
        return this.splittersEnabled.get(PF2);
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier + "\n[CLASSIFIER SUMMARY] splitterConfig: " + splitterConfig;
    }
}
