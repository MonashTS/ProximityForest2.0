package application;


import java.io.File;
import java.util.Random;

public class Application {
    public static final String optionsSep = "=";
    public static final String paramSep = ":";
    public static String outputPath;
    public static String datasetPath;
    public static String problem = "";
    public static String classifierName = "DTW-1NN";
    public static String machine = "windows";
    public static int paramId = 0;
    public static int verbose = 0;
    public static boolean retrain = true;
    public static int numThreads = 0;
    public static int iteration = 0;
    public static int ueaFold = 0;
    public static int fold8020 = 0;
    public static long seed = System.nanoTime();
    public static double sampleSize = 1;
    public static boolean doEvaluation = true;
    private final static String defaultSaveFilename = "results.csv";
    private final static String defaultSaveFilenameJSON = "results.json";

    public static Random rand;

    public static void setDefaultDatasetPath() {
        if (datasetPath == null) {
            String username = System.getProperty("user.name");
            switch (machine) {
                case "windows":
                    datasetPath = "C:/Users/" + username + "/workspace/Dataset/UCRArchive_2018/";
                    break;
                case "m3":
                    datasetPath = "/projects/nc23/changwei/Dataset/UCRArchive_2018/";
                    break;
                case "wsl":
                    datasetPath = "/mnt/c/Users/" + username + "/workspace/Dataset/UCRArchive_2018/";
                    break;
                default:
                    datasetPath = "/home/" + username + "/workspace/Dataset/UCRArchive_2018/";
                    break;
            }
        }
    }

    public static boolean isDatasetDone(String outputPath) {
        // check that output files exist
        File f1 = new File(outputPath + defaultSaveFilename);
        File f2 = new File(outputPath + defaultSaveFilenameJSON);
        return f1.exists() && !f1.isDirectory() && f2.exists() && !f2.isDirectory();
    }

    public static void printSummary(String moduleName) {
        System.out.println("[" + moduleName + "] Machine: " + Application.machine);
        System.out.println("[" + moduleName + "] DatasetPath: " + Application.datasetPath);
        System.out.println("[" + moduleName + "] Problem: " + Application.problem);
        System.out.println("[" + moduleName + "] Classifier: " + Application.classifierName);
        System.out.println("[" + moduleName + "] ParamId: " + Application.paramId);
        System.out.println();
    }

    public static void setOutputPath() {
        if (Application.outputPath == null) {
            String iterNum;
            if (Application.fold8020 > 0) iterNum = "resamples_" + Application.fold8020;
            else if (Application.ueaFold > 0) iterNum = "fold_" + Application.ueaFold;
            else iterNum = Application.iteration + "";

            if (Application.paramId > 0)
                // probably only applicable to NN
                Application.outputPath = System.getProperty("user.dir") + "/outputs/benchmark/" + Application.classifierName + "_" + Application.paramId + "/" + iterNum + "/" + problem + "/";
            else
                Application.outputPath = System.getProperty("user.dir") + "/outputs/benchmark/" + Application.classifierName + "/" + iterNum + "/" + problem + "/";
        }
    }
}
