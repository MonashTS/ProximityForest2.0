package results;

import utils.Tools;

public class PredictionResults {
    public int[] index;
    public int nCorrect;
    public double accuracy;
    public int[] predictions;
    public int[][] classCounts;
    public double[][] classProbas;
    public long elapsedTimeNanoSeconds;
    public int paramId;
    public String paramStr;
    public int[] cvParams;
    public double[] cvAcc;
    public String trainTest = "";

    public PredictionResults() {
        this.accuracy = 0;
        this.elapsedTimeNanoSeconds = 0;
    }

    public PredictionResults(int[] index, int nCorrect, int[] predictions, int[][] classCounts) {
        this.index = index;
        this.nCorrect = nCorrect;
        this.predictions = predictions;
        this.classCounts = classCounts;
    }

    public PredictionResults(int nCorrect, double acc, int[] predictions, int[][] classCounts) {
        this.nCorrect = nCorrect;
        this.accuracy = acc;
        this.predictions = predictions;
        this.classCounts = classCounts;
    }

    public PredictionResults(int[] index, int nCorrect, int[] predictions, int[][] classCounts, String trainTest) {
        this.index = index;
        this.nCorrect = nCorrect;
        this.predictions = predictions;
        this.classCounts = classCounts;
        this.trainTest = trainTest;
    }

    public PredictionResults(int nCorrect, double acc, int[] predictions, int[][] classCounts, String trainTest) {
        this.nCorrect = nCorrect;
        this.accuracy = acc;
        this.predictions = predictions;
        this.classCounts = classCounts;
        this.trainTest = trainTest;
    }

    public void setTime(long startTimeNano, long stopTimeNano) {
        this.elapsedTimeNanoSeconds = stopTimeNano - startTimeNano;
    }


    public void setParamId(int paramId) {
        this.paramId = paramId;
    }

    public void setParamStr(String paramStr) {
        this.paramStr = paramStr;
    }

    public void setTrainTest(String s) {
        this.trainTest = s;
    }

    public void setCvAcc(double[] cvAcc) {
        this.cvAcc = cvAcc;
    }

    public void setCvParams(int[] cvParams) {
        this.cvParams = cvParams;
    }

    public void calcClassProbas() {
        this.classProbas = new double[classCounts.length][classCounts[0].length];
        for (int i = 0; i < classCounts.length; i++) {
            double sum = 0;
            for (int j = 0; j < classCounts[i].length; j++) {
                sum += classCounts[i][j];
            }
            for (int j = 0; j < classCounts[i].length; j++) {
                this.classProbas[i][j] = 1.0 * classCounts[i][j] / sum;
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder probas;
        if (classCounts != null) {
            if (classProbas == null) calcClassProbas();
            probas = new StringBuilder("\"[");
            for (int i = 0; i < classProbas.length; i++) {
                probas.append("[").append(classProbas[i][0]);
                for (int j = 1; j < classProbas[i].length; j++) {
                    probas.append(",").append(classProbas[i][j]);
                }
                probas.append("]");
                if (i < classProbas.length - 1) probas.append(",");
            }
            probas.append("]\"");
        } else {
            probas = new StringBuilder("");
        }

        StringBuilder preds;
        if (predictions != null) {
            preds = new StringBuilder("\"[");
            preds.append(predictions[0]);
            for (int i = 1; i < predictions.length; i++) {
                preds.append(",").append(predictions[i]);
            }
            preds.append("]\"");
        } else {
            preds = new StringBuilder("");
        }

        StringBuilder cvParamsStr;
        if (cvParams != null) {
            cvParamsStr = new StringBuilder("\"[");
            cvParamsStr.append(cvParams[0]);
            for (int i = 1; i < cvParams.length; i++) {
                cvParamsStr.append(",").append(cvParams[i]);
            }
            cvParamsStr.append("]\"");
        } else {
            cvParamsStr = null;
        }

        StringBuilder cvAccStr;
        if (cvAcc != null) {
            cvAccStr = new StringBuilder("\"[");
            cvAccStr.append(cvAcc[0]);
            for (int i = 1; i < cvParams.length; i++) {
                cvAccStr.append(",").append(cvAcc[i]);
            }
            cvAccStr.append("]\"");
        } else {
            cvAccStr = null;
        }

        if (!trainTest.equals("")) trainTest = trainTest + "_";
        String ss = paramStr;
        if (paramStr == null) ss = "";
        else if (!paramStr.equals("")) ss = ss + ",";

        String s = "\"" + trainTest + "duration\":" + (elapsedTimeNanoSeconds / 1e9) + "," +
                "\"" + trainTest + "acc\":" + accuracy + "," +
                "\"" + trainTest + "nb_correct\":" + nCorrect + "," +
                "\"param_id\":" + paramId + "," +
                ss +
                "\"" + trainTest + "predictions\":" + preds + "," +
                "\"" + trainTest + "proba\":" + probas;
        if (cvParamsStr != null) s = s + ",\"cv_params\":" + cvParamsStr;
        if (cvAccStr != null) s = s + ",\"cv_accs\":" + cvAccStr;
        return "{" + s + "}";
    }

    public String doTime() {
        return Tools.doTime(this.elapsedTimeNanoSeconds);
    }
}
