package datasets;

import application.Application;
import distances.ED;
import utils.Tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

public class Sequences {
    public ArrayList<Sequence> data;
    public int[] indices;
    public double std = -1;
    public HashMap<Double, double[]> adtwWeights;
    public int[] lcssDeltas;
    public double[] lcssEpsilons;
    public boolean epsilonsAndDeltasRefreshed;

    protected HashMap<Integer, Integer> classDistribution;
    protected HashMap<Integer, ArrayList<Integer>> classIndex;

    public Sequences() {
        this.data = new ArrayList<>();
    }

    public Sequences(final int dataSize) {
        this.data = new ArrayList<>(dataSize);
    }

    public Sequences(final ArrayList<Sequence> x) {
        this.data = x;
        this.setClassDistribution();
    }

    public Sequences(final ArrayList<Sequence> x, final int[] index) {
        this.data = x;
        this.indices = index;
        this.setClassDistribution();
    }

    public void add(final Sequence s, final int ix) {
        if (this.data == null) {
            this.data = new ArrayList<>();
        }
        this.data.add(s);
        if (this.classDistribution == null) {
            this.classDistribution = new HashMap<>();
            this.classIndex = new HashMap<>();
        }
        if (!this.classDistribution.containsKey(s.classLabel)) {
            this.classIndex.put(s.classLabel, new ArrayList<>());
            this.classDistribution.put(s.classLabel, 0);
        }
        this.classIndex.get(s.classLabel).add(ix);
        this.classDistribution.put(s.classLabel, this.classDistribution.get(s.classLabel) + 1);
    }

    public Sequence get(final int i) {
        if (i < this.size())
            return this.data.get(i);
        return null;
    }

    public int length() {
        return data.get(0).length();
    }

    public int dim() {
        return data.get(0).get().length;
    }

    public void shuffle() {
        Collections.shuffle(this.data, new Random(Application.iteration));
    }

    public int size() {
        return this.data.size();
    }

    public int[] getLabels() {
        final int[] labels = new int[this.size()];
        for (int i = 0; i < this.size(); i++)
            labels[i] = this.get(i).classLabel;

        return labels;
    }

    public void setLabels(final int[] labels) {
        for (int i = 0; i < this.size(); i++)
            this.get(i).setLabel(labels[i]);
    }

    public static Sequences stratifySubset(final Sequences X, final double percent) {
        final int trainSize = X.size();
        int sampleSize = (int) (percent * trainSize);

        final HashMap<Integer, ArrayList<Integer>> classMapping = new HashMap<>(trainSize);
        for (int i = 0; i < trainSize; i++) {
            final int a = X.get(i).classLabel;
            if (!classMapping.containsKey(a)) {
                classMapping.put(a, new ArrayList<>());
            }
            classMapping.get(a).add(i);
        }
        for (Integer c : classMapping.keySet())
            Collections.shuffle(classMapping.get(c), new Random(trainSize + sampleSize + X.length()));

        final int nClass = classMapping.size();
        sampleSize = Math.max(sampleSize, nClass);
        int c = 0;
        int i = 0;
        int outSize = 0;

        final ArrayList<Sequence> xOut = new ArrayList<>(sampleSize);
        final int[] index = new int[sampleSize];
        while (outSize < sampleSize || i < 0) {
            if (classMapping.get(c).size() > i) {
                final int idx = classMapping.get(c).get(i);
                xOut.add(X.get(idx));
                index[outSize] = idx;
                outSize++;
            }
            c++;
            if (c == nClass) {
                c = 0;
                i++;
            }
        }

        return new Sequences(xOut, index);
    }

    public void setClassDistribution() {
        ArrayList<Integer> temp;
        this.classDistribution = new HashMap<>();
        this.classIndex = new HashMap<>();
        for (int i = 0; i < this.data.size(); i++) {
            final int label = this.get(i).classLabel;

            if (classDistribution.containsKey(label)) {
                classDistribution.put(label, classDistribution.get(label) + 1);
                temp = classIndex.get(label);
            } else {
                classDistribution.put(label, 1);
                temp = new ArrayList<>();
            }
            temp.add(i);
            classIndex.put(label, temp);
        }
    }

    public int getNumClasses() {
        if (this.classDistribution == null)
            setClassDistribution();

        return this.classDistribution.size();
    }

    public HashMap<Integer, Sequences> splitByClass() {
        final HashMap<Integer, Sequences> split = new HashMap<>(this.getNumClasses());

        for (Integer label : this.classDistribution.keySet()) {
            int numInstances = this.classDistribution.get(label);
            ArrayList<Sequence> x = new ArrayList<>(numInstances);
            int[] index = new int[numInstances];
            ArrayList<Integer> temp = classIndex.get(label);
            for (int i = 0; i < numInstances; i++)
                index[i] = temp.get(i);
            split.put(label, new Sequences(x, index));
        }

        return split;
    }

    public HashMap<Integer, Integer> getClassDistribution() {
        return this.classDistribution;
    }

    public double getStd_p() {
        if (this.std < 0)
            this.std = Tools.stdv_p(this);
        return this.std;
    }

    public double[] initADTWWeights(double gammaExponent) {
        return initADTWWeights(4000, 100, 5, gammaExponent);
    }

    public double[] initADTWWeights(int nSamples, int nParams, int exponent, double gammaExponent) {
        if (adtwWeights == null)
            adtwWeights = new HashMap<>();

        if (!adtwWeights.containsKey(gammaExponent)) {
            int maxWeight = 0;
            final int trainSize = this.size();
            double[][] pairDist = new double[trainSize][trainSize];
            Random random = new Random(trainSize + this.length());
            for (int i = 0; i < nSamples; i++) {
                int a = random.nextInt(trainSize);
                int b = random.nextInt(trainSize);
                while (a == b) b = random.nextInt(trainSize);
                if (pairDist[a][b] == 0) {
                    double dist = ED.distanceGe(this.get(a).firstChannel(), this.get(b).firstChannel(), gammaExponent);
                    pairDist[a][b] = dist;
                    pairDist[b][a] = dist;
                }
                maxWeight += pairDist[a][b];
            }
            maxWeight /= nSamples;

            final double[] w = new double[nParams];
            for (int i = 1; i <= nParams; i++) w[i - 1] = maxWeight * Math.pow(i * 0.01, exponent);
            adtwWeights.put(gammaExponent, w);
        }


        return adtwWeights.get(gammaExponent);
    }

    public double[] initADTWWeights(int nSamples, int nParams, int exponent) {
        if (adtwWeights == null)
            adtwWeights = new HashMap<>();

        if (!adtwWeights.containsKey(2.0)) {
            int maxWeight = 0;
            final int trainSize = this.size();
            double[][] pairDist = new double[trainSize][trainSize];
            Random random = new Random(trainSize + this.length());
            for (int i = 0; i < nSamples; i++) {
                int a = random.nextInt(trainSize);
                int b = random.nextInt(trainSize);
                while (a == b) b = random.nextInt(trainSize);
                if (pairDist[a][b] == 0) {
                    double dist = ED.distance(this.get(a).firstChannel(), this.get(b).firstChannel());
                    pairDist[a][b] = dist;
                    pairDist[b][a] = dist;
                }
                maxWeight += pairDist[a][b];
            }
            maxWeight /= nSamples;

            final double[] w = new double[nParams];
            for (int i = 1; i <= nParams; i++) w[i - 1] = maxWeight * Math.pow(i * 0.01, exponent);
            adtwWeights.put(2.0, w);
        }

        return adtwWeights.get(2.0);
    }

    public void initLCSSParam() {
        if (!this.epsilonsAndDeltasRefreshed) {
            final double stdTrain = this.getStd_p();
            final double stdFloor = stdTrain * 0.2;
            this.lcssEpsilons = Tools.getInclusive10(stdFloor, stdTrain);
            this.lcssDeltas = Tools.getInclusive10(0, (this.length() + 1) / 4);
            this.epsilonsAndDeltasRefreshed = true;
        }
    }
}
