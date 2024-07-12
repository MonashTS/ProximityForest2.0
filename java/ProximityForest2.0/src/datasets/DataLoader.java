package datasets;

import application.Application;
import processors.missingValuesProcessors.LinearInterpolator;
import processors.missingValuesProcessors.MissingValuesProcessor;
import processors.variableLengthProcessors.SuffixNoisePadder;
import processors.variableLengthProcessors.VaryLengthProcessor;
import utils.Tools;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class DataLoader {
    // processors
    private final MissingValuesProcessor missingValuesProcessor = new LinearInterpolator();
    private final VaryLengthProcessor varyLengthProcessor = new SuffixNoisePadder();

    private int[] getCSVFileInformation(final String fileName, boolean hasHeader, final String fileDelimiter) throws IOException {
        final FileReader input = new FileReader(fileName);
        final LineNumberReader lineNumberReader = new LineNumberReader(input);
        String line;
        String[] lineArray = null;
        final int[] fileInfo = new int[2];

        try {
            boolean lengthCheck = true;

            while ((line = lineNumberReader.readLine()) != null) {
                if (lengthCheck) {
                    lengthCheck = false;
                    lineArray = line.split(fileDelimiter);
                }
            }
        } finally {
            input.close();
        }

        //this output array contains file information
        //number of rows;
        if (hasHeader)
            fileInfo[0] = lineNumberReader.getLineNumber() == 0 ? lineNumberReader.getLineNumber() : lineNumberReader.getLineNumber() - 1;
        else
            fileInfo[0] = lineNumberReader.getLineNumber();


        assert lineArray != null;
        fileInfo[1] = lineArray.length;  //number of columns;

        return fileInfo;
    }

    public Sequences readMonster(final String datasetName, final String datasetPath) {
        String xPath = datasetPath + datasetName + "/" + datasetName + "_X.csv";
        String yPath = datasetPath + datasetName + "/" + datasetName + "_y.csv";
        String metaPath = datasetPath + datasetName + "/metadata/" + datasetName + "_metadata.csv";

        return readCSVFileToSequences(xPath, yPath, metaPath, ",");
    }

    public Sequences readUCRTrain(final String datasetName, final String datasetPath) {
        String path = datasetPath + datasetName + "/" + datasetName + "_TRAIN.tsv";
        return readTSVFileToSequences(path, true);
    }

    public Sequences readUCRTest(final String datasetName, final String datasetPath) {
        String path = datasetPath + datasetName + "/" + datasetName + "_TEST.tsv";
        return readTSVFileToSequences(path, true);
    }

    public Sequences readResampledUCRTrain(final String datasetName, final String datasetPath, final int fold) {
        final String path = datasetPath + "UCR_Resamples/resampletrain" + fold + "/TRAIN_RESAMPLE_" + fold + "_" + datasetName + ".txt";
        File f1 = new File(path);
        if (!f1.exists()) return null;

        return readCSVFileToSequences(path, true, " ");
    }

    public Sequences readResampledUCRTest(final String datasetName, final String datasetPath, final int fold) {
        final String path = datasetPath + "UCR_Resamples/resampletest" + fold + "/TEST_RESAMPLE_" + fold + "_" + datasetName + ".txt";
        File f1 = new File(path);
        if (!f1.exists()) return null;

        return readCSVFileToSequences(path, true, " ");
    }

    public Sequences readTSVFileToSequences(final String fileName, boolean targetColumnIsFirst) {
        return readCSVFileToSequences(fileName, targetColumnIsFirst, "\t");
    }

    public ArrayList<Sequence> readCSVFileToSequences(final String fileName, boolean targetColumnIsFirst) {
        return readCSVFileToSequences(fileName, targetColumnIsFirst);
    }

    public Sequences readCSVFileToSequences(final String fileName, boolean targetColumnIsFirst, final String fileDelimiter) {
        String line;
        String[] lineArray;

        int label;
        int[] fileInfo;
        final File f = new File(fileName);
        boolean hasMissing = false;

        BufferedReader br = null;
        Sequences dataset = null;
        try {
            if (Application.verbose > 1) System.out.print("[DATASET-LOADER] reading [" + f.getName() + "]: ");
            final long startTime = System.nanoTime();

            // useful for reading large files;
            fileInfo = getCSVFileInformation(fileName, false, fileDelimiter); // 0=> no. of rows 1=> no. columns
            final int expectedSize = fileInfo[0];
            final int seriesLength = fileInfo[1] - 1;  //-1 to exclude target the column

            // initialise
            br = new BufferedReader(new FileReader(fileName));
            dataset = new Sequences(expectedSize);

            int count = 0;
            while ((line = br.readLine()) != null) {
                lineArray = line.split(fileDelimiter);

                double[] tmp = new double[seriesLength];

                // read the data
                if (targetColumnIsFirst) {
                    for (int j = 1; j <= seriesLength; j++) {
                        tmp[j - 1] = Double.parseDouble(lineArray[j]);
                        if (Tools.isMissing(tmp[j - 1])) hasMissing = true;
                    }
                    label = Integer.parseInt(lineArray[0]);
                } else {
                    int j;
                    for (j = 0; j < seriesLength; j++) {
                        tmp[j] = Double.parseDouble(lineArray[j]);
                        if (Tools.isMissing(tmp[j])) hasMissing = true;
                    }
                    label = Integer.parseInt(lineArray[j]);
                }

                if (Tools.isMissing(tmp[tmp.length - 1]))
                    tmp = varyLengthProcessor.process(tmp, seriesLength);

                if (hasMissing && !Tools.isMissing(tmp[tmp.length - 1]))
                    tmp = missingValuesProcessor.process(tmp);

                dataset.add(new Sequence(tmp, label), count);
                count++;
            }
            final long endTime = System.nanoTime();
            final long elapsed = endTime - startTime;
            final String timeDuration = Tools.doTime(1.0 * elapsed / 1e6);
            if (Application.verbose > 1) System.out.println(" finished in " + timeDuration);

            // reorder class
            if (Application.iteration > 0) dataset.shuffle();
        } catch (IOException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return dataset;
    }

    public Sequences readCSVFileToSequences(final String xFile, final String yFile, final String metaFile, final String fileDelimiter) {
        String line;
        String[] lineArray;

        int label;
        int[] fileInfo;
        final File fmeta = new File(metaFile);
        final File fX = new File(xFile);
        final File fy = new File(yFile);
        boolean hasMissing = false;

        BufferedReader br = null;
        Sequences dataset = null;
        ArrayList<Integer> classLabels = new ArrayList<>();
        int nInstances = 0;
        int nDim = 0;
        int seqlen = 0;

        // read in the metadata
        try {
            if (Application.verbose > 1) System.out.print("[DATASET-LOADER] reading [" + fmeta.getName() + "]: ");
            final long startTime = System.nanoTime();
            // initialise
            br = new BufferedReader(new FileReader(xFile));

            while ((line = br.readLine()) != null) {
                lineArray = line.split(fileDelimiter);
                System.out.println(Arrays.toString(lineArray));
                if (lineArray[0].equals("n_instances")) nInstances = Integer.parseInt(lineArray[1]);
                else if (lineArray[0].equals("n_dim")) nDim = Integer.parseInt(lineArray[1]);
                else if (lineArray[0].equals("series_length")) seqlen = Integer.parseInt(lineArray[1]);
            }
            final long endTime = System.nanoTime();
            final long elapsed = endTime - startTime;
            final String timeDuration = Tools.doTime(1.0 * elapsed / 1e6);
            if (Application.verbose > 1) {
                System.out.println(" finished in " + timeDuration);
                System.out.println(nInstances + " instances, " + nDim + " dimensions, " + seqlen + " long");
            }

        } catch (IOException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        // read in class labels
        try {
            if (Application.verbose > 1) System.out.print("[DATASET-LOADER] reading [" + fy.getName() + "]: ");
            final long startTime = System.nanoTime();

            // initialise
            br = new BufferedReader(new FileReader(yFile));

            while ((line = br.readLine()) != null) {
                lineArray = line.split(fileDelimiter);
                label = Integer.parseInt(lineArray[0]);
                classLabels.add(label);
            }
            final long endTime = System.nanoTime();
            final long elapsed = endTime - startTime;
            final String timeDuration = Tools.doTime(1.0 * elapsed / 1e6);
            if (Application.verbose > 1) System.out.println(" finished in " + timeDuration);
        } catch (IOException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        // read the X
        try {
            if (Application.verbose > 1) System.out.print("[DATASET-LOADER] reading [" + fX.getName() + "]: ");
            final long startTime = System.nanoTime();

            // initialise
            br = new BufferedReader(new FileReader(xFile));
            dataset = new Sequences(nInstances);

            int count = 0;
            while ((line = br.readLine()) != null) {
                lineArray = line.split(fileDelimiter);

                if (nDim == 1) {
                    double[] tmp = new double[seqlen];
                    for (int j = 0; j < seqlen; j++) {
                        tmp[j] = Double.parseDouble(lineArray[j]);
                        if (Tools.isMissing(tmp[j])) hasMissing = true;
                    }
                    if (Tools.isMissing(tmp[tmp.length - 1]))
                        tmp = varyLengthProcessor.process(tmp, seqlen);

                    if (hasMissing && !Tools.isMissing(tmp[tmp.length - 1]))
                        tmp = missingValuesProcessor.process(tmp);

                    dataset.add(new Sequence(tmp, classLabels.get(count)), count);
                } else {
                    double[][] tmp = new double[nDim][seqlen];
                    int jj = 0;
                    for (int k = 0; k < nDim; k++) {
                        for (int j = 0; j < seqlen; j++) {
                            tmp[k][j] = Double.parseDouble(lineArray[jj]);
                            if (Tools.isMissing(tmp[k][j])) hasMissing = true;
                            jj++;
                        }
                        if (Tools.isMissing(tmp[k][tmp.length - 1]))
                            tmp[k] = varyLengthProcessor.process(tmp[k], seqlen);
                        if (hasMissing && !Tools.isMissing(tmp[k][tmp.length - 1]))
                            tmp[k] = missingValuesProcessor.process(tmp[k]);
                    }

                    dataset.add(new Sequence(tmp, classLabels.get(count)), count);
                }

                count++;
            }
            final long endTime = System.nanoTime();
            final long elapsed = endTime - startTime;
            final String timeDuration = Tools.doTime(1.0 * elapsed / 1e6);
            if (Application.verbose > 1) System.out.println(" finished in " + timeDuration);

            // reorder class
            if (Application.iteration > 0) dataset.shuffle();
        } catch (IOException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return dataset;
    }

    public Sequences[] ueaResamples(final String problem, final String datasetPath, final Sequences train, final Sequences test, final int fold) {
        String path = datasetPath + problem + "_INDICES_TRAIN.txt";
        int[] trainIndices = loadUEAResamples(path, fold);
        path = datasetPath + problem + "_INDICES_TEST.txt";
        int[] testIndices = loadUEAResamples(path, fold);

        Sequences newTrain = new Sequences(trainIndices.length);
        Sequences newTest = new Sequences(testIndices.length);

        for (int i = 0; i < trainIndices.length; i++) {
            int ii = trainIndices[i];
            if (ii < train.size()) {
                // new data from train
                newTrain.add(train.get(ii), i);
            } else {
                // new data from test
                newTrain.add(test.get(ii - train.size()), i);
            }
        }
        for (int i = 0; i < testIndices.length; i++) {
            int ii = testIndices[i];
            if (ii < train.size()) {
                // new data from train
                newTest.add(train.get(ii), i);
            } else {
                // new data from test
                newTest.add(test.get(ii - train.size()), i);
            }
        }
        return new Sequences[]{newTrain, newTest};
    }

    public int[] loadUEAResamples(final String fileName, final int fold) {
        BufferedReader br = null;
        int[] indices = null;
        String line;
        String[] lineArray;

        int count;
        int[] fileInfo;
        final File f = new File(fileName);
        String fileDelimiter = " ";

        try {
            if (Application.verbose > 1) System.out.print("[DATASET-LOADER] reading [" + f.getName() + "]: ");
            final long startTime = System.nanoTime();

            // useful for reading large files;
            fileInfo = getCSVFileInformation(fileName, false, fileDelimiter); // 0=> no. of rows 1=> no. columns
            final int indicesLength = fileInfo[1];  //-1 to exclude target the column

            // initialise
            indices = new int[indicesLength];

            count = 0;
            br = new BufferedReader(new FileReader(fileName));

            while ((line = br.readLine()) != null) {
                if (count == fold) {
                    lineArray = line.split(fileDelimiter);
                    for (int j = 0; j < indicesLength; j++) {
                        indices[j] = Integer.parseInt(lineArray[j]);
                    }
                }
                count++;
            }
            final long endTime = System.nanoTime();
            final long elapsed = endTime - startTime;
            final String timeDuration = Tools.doTime(1.0 * elapsed / 1e6);
            if (Application.verbose > 1) System.out.println(" finished in " + timeDuration);
        } catch (IOException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return indices;
    }

    public static Sequences[] normaliseLabels(Sequences train, Sequences test) {
        int[] yTrain = train.getLabels();
        int[] yTest = test.getLabels();

        int classCount = 0;
        HashMap<Integer, Integer> classMapping = new HashMap<>();
        for (int i = 0; i < train.size(); i++) {
            final Sequence s = train.get(i);
            if (!classMapping.containsKey(s.classLabel)) {
                classMapping.put(s.classLabel, classCount);
                classCount++;
            }
        }
        for (int i = 0; i < test.size(); i++) {
            final Sequence s = test.get(i);
            if (!classMapping.containsKey(s.classLabel)) {
                classMapping.put(s.classLabel, classCount);
                classCount++;
            }
        }
        for (int i = 0; i < yTrain.length; i++) yTrain[i] = classMapping.get(yTrain[i]);
        for (int i = 0; i < yTest.length; i++) yTest[i] = classMapping.get(yTest[i]);

        train.setLabels(yTrain);
        test.setLabels(yTest);

        return new Sequences[]{train, test};
    }
}
