package application;

import datasets.DataLoader;
import datasets.Sequences;
import results.PredictionResults;
import tree.ProximityForest;
import tree.results.PFResults;
import tree.splitters.NodeSplitter;
import utils.OutFile;
import utils.Tools;

import java.util.HashMap;
import java.util.Random;

import static tree.splitters.NodeSplitter.SplitterType.PF;
import static tree.splitters.NodeSplitter.SplitterType.PF2;
import static utils.Tools.doTimeNs;

public class MonsterClassification {
    final static HashMap<NodeSplitter.SplitterType, Integer> candidates = new HashMap<>();
    private static final String[] testArgs = new String[]{
            "-data=C:/Users/cwtan/workspace/Dataset/UCRArchive_2018/",
            "-problem=ArrowHead",                                   // dataset name
            "-cpu=4",                                               // number of cpu cores/threads
            "-verbose=1",                                           // verbosity, 0, 1, 2
            "-iter=0",                                              // iteration runs
            "-fold=0",                                              // 5 Monash 80/20 splits
            "-eval=true",                                           // to evaluate or not
            "-param=numTrees:100,n:0.1,pf2:5", "-seed=1234"};

    static String moduleName = "MonsterClassification";
    static int numTrees = 100;
    static double nSamples = 0;
    static int pfCandidates = 0;
    static int pf2Candidates = 0;

    private static void extractArguments(final String[] args) throws Exception {
        System.out.print("[APP] Input arguments:");
        for (String arg : args) {
            final String[] options = arg.trim().split(Application.optionsSep);
            System.out.print(" " + arg);
            if (options.length >= 2) switch (options[0]) {
                case "-out":
                    Application.outputPath = options[1];
                    break;
                case "-machine":
                    Application.machine = options[1];
                    break;
                case "-data":
                    Application.datasetPath = options[1];
                    break;
                case "-problem":
                    Application.problem = options[1];
                    break;
                case "-param":
                    final String[] paramOpts = options[1].trim().split(",");
                    for (String paramOpt : paramOpts) {
                        final String[] s = paramOpt.trim().split(Application.paramSep);
                        if (s.length >= 2) {
                            switch (s[0]) {
                                case "numTrees":
                                    numTrees = Integer.parseInt(s[1]);
                                    break;
                                case "n":
                                    nSamples = Double.parseDouble(s[1]);
                                    break;
                                case "pf":
                                    pfCandidates = Integer.parseInt(s[1]);
                                    break;
                                case "pf2":
                                    pf2Candidates = Integer.parseInt(s[1]);
                                    break;
                            }
                        }
                    }
                    break;
                case "-cpu":
                    Application.numThreads = Integer.parseInt(options[1]);
                    if (Application.numThreads < 0) Application.numThreads = Runtime.getRuntime().availableProcessors();
                    else
                        Application.numThreads = Math.min(Application.numThreads, Runtime.getRuntime().availableProcessors());
                    Application.numThreads = Math.max(Application.numThreads, 1);
                    break;
                case "-iter":
                    Application.iteration = Integer.parseInt(options[1]);
                    break;
                case "-fold":
                    Application.fold8020 = Integer.parseInt(options[1]);
                    break;
                case "-verbose":
                    Application.verbose = Integer.parseInt(options[1]);
                    break;
                case "-eval":
                    Application.doEvaluation = Boolean.parseBoolean(options[1]);
                    break;
                case "-seed":
                    Application.seed = Long.parseLong(options[1]);
                    break;
                default:
                    throw new Exception("Try -out <output_path>, -data <dataset_path>, -problem <problem>, -paramId <paramId>");
            }
            else
                throw new Exception("Try -out <output_path>, -data <dataset_path>, -problem <problem>, -paramId <paramId>");
        }

        System.out.println();

        Application.rand = new Random(Application.seed);
        Application.classifierName = "ProximityForest 2.0";
        Application.paramId = 0;
        Application.setOutputPath();
        Application.setDefaultDatasetPath();
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) args = testArgs;
        extractArguments(args);

        // print a summary of the run before the experiments
        Application.printSummary(moduleName);

        // start the experiment
        final long startTime = System.nanoTime();
        singleRun(Application.problem);
        final long elapsedTime = System.nanoTime() - startTime;
        System.out.println("[" + moduleName + "] Total time taken " + doTimeNs(elapsedTime));
    }

    /**
     * Single run of the experiments
     */
    private static void singleRun(String problem) throws Exception {
        Application.setOutputPath();
        System.out.println(Application.outputPath);

        if (!Application.retrain && Application.isDatasetDone(Application.outputPath)) return;

        // load data
        final DataLoader loader = new DataLoader();
        Sequences trainData;
        Sequences testData;

        {
            Sequences[] data = loader.readMonster(problem, Application.datasetPath, Application.fold8020);
            trainData = data[0];
            testData = data[1];
        }

        // initialising the classifier
        candidates.put(PF, pfCandidates);
        candidates.put(PF2, pf2Candidates);

        if (nSamples <= 0) nSamples = 1.0;
        if (trainData.size() < 10000) nSamples = 1.0;

        final ProximityForest classifier = new ProximityForest(numTrees, candidates, nSamples);
        classifier.setThreads(Application.numThreads);
        System.out.println(classifier);
        System.out.println("[" + moduleName + "] Training data: (" + trainData.size() + "," + trainData.length() + "," + trainData.dim() + ")");
        System.out.println("[" + moduleName + "] Test data: (" + testData.size() + "," + testData.length() + "," + testData.dim() + ")");

        // training the classifier
        PredictionResults trainingResults;
        if (Application.sampleSize == 1) trainingResults = classifier.fit(trainData);
        else trainingResults = classifier.fit(trainData, Application.sampleSize);

        System.out.println("[" + moduleName + "]" + trainingResults);
        System.out.println(classifier);

        // do the evaluation
        double totalTime = trainingResults.elapsedTimeNanoSeconds;
        if (Application.doEvaluation) {
            PFResults classificationResults = classifier.evaluate(testData);
            System.out.println("[" + moduleName + "]" + classificationResults);
            totalTime += classificationResults.elapsedTimeNanoSeconds;

            saveResults(
                    Application.outputPath,
                    problem, Application.classifierName, Application.numThreads,
                    (PFResults) trainingResults, classificationResults,
                    "results_" + nSamples + ".csv");
        }
        System.out.println("[" + moduleName + "] Total time taken " + Tools.doTime(totalTime));
    }

    private static void saveResults(String outputPath, String problem, String classifier, int nThreads, PFResults trainResults, PFResults testResults, String filename) throws Exception {
        final OutFile outFile = new OutFile(outputPath, filename);
        outFile.writeLine("problem," + problem);
        outFile.writeLine("classifier," + classifier);
        outFile.writeLine("nThreads," + nThreads);
        outFile.writeLine("n_samples," + nSamples);
        outFile.writeLine("pf_candidates," + pfCandidates);
        outFile.writeLine("pf2_candidates," + pf2Candidates);

        outFile.writeLine("accuracy," + testResults.accuracy);
        outFile.writeLine("nb_correct," + testResults.nCorrect);
        outFile.writeLine("train_parallel_time," + trainResults.doTime());
        outFile.writeLine("train_parallel_time_ns," + trainResults.elapsedTimeNanoSeconds);
        outFile.writeLine("train_serial_time," + Tools.doTimeNs(trainResults.totalTrainTime));
        outFile.writeLine("train_serial_time_ns," + trainResults.totalTrainTime);
        outFile.writeLine("test_parallel_time," + testResults.doTime());
        outFile.writeLine("test_parallel_time_ns," + testResults.elapsedTimeNanoSeconds);
        outFile.writeLine("test_serial_time," + Tools.doTimeNs(testResults.totalTestTime));
        outFile.writeLine("test_serial_time_ns," + testResults.totalTestTime);

        outFile.writeLine("train_hydra_transform_time_ns," + trainResults.hydraTransformTime);
        outFile.writeLine("test_hydra_transform_time_ns," + testResults.hydraTransformTime);

        outFile.writeLine("train_pf_time_ns," + trainResults.splitterTime.get(PF));
        outFile.writeLine("train_pf2_time_ns," + trainResults.splitterTime.get(PF2));

        outFile.writeLine("test_pf_time_ns," + testResults.splitterTime.get(PF));
        outFile.writeLine("test_pf2_time_ns," + testResults.splitterTime.get(PF2));

        outFile.writeLine("leaf_count," + trainResults.leafCount);
        outFile.writeLine("pf_count," + trainResults.splitterCount.get(PF));
        outFile.writeLine("pf2_count," + trainResults.splitterCount.get(PF2));

        testResults.calcClassProbas();
        StringBuilder str = new StringBuilder("predictions\n");
        for (int i = 0; i < testResults.predictions.length; i++) {
            str.append(testResults.predictions[i]).append(",");
            str.append(testResults.classProbas[i][0]);
            for (int j = 1; j < testResults.classProbas[i].length; j++) {
                str.append(",").append(testResults.classProbas[i][j]);
            }
            str.append("\n");
        }
        outFile.writeLine(str.toString());

        outFile.closeFile();
    }
}

