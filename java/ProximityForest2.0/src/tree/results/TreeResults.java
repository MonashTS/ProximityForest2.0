package tree.results;

import results.PredictionResults;

public class TreeResults extends PredictionResults {
    public double totalTrainTime = 0;
    public double totalTestTime = 0;

    public TreeResults() {
        super();
    }

    public TreeResults(final int[] index, final int nCorrect, final int[] predictions, final int[][] classCounts) {
        super(index, nCorrect, predictions, classCounts);
    }

    public TreeResults(final int nCorrect, final double acc, final int[] predictions, final int[][] classCounts) {
        super(nCorrect, acc, predictions, classCounts);
    }

    public TreeResults(final int[] index, final int nCorrect, final int[] predictions, final int[][] classCounts, final String trainTest) {
        super(index, nCorrect, predictions, classCounts, trainTest);
    }

    public TreeResults(final int nCorrect, final double acc, final int[] predictions, final int[][] classCounts, final String trainTest) {
        super(nCorrect, acc, predictions, classCounts, trainTest);
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

        String s = "\"" + trainTest + "duration\":" + elapsedTimeNanoSeconds + "," +
                "\"" + trainTest + "acc\":" + accuracy + "," +
                "\"" + trainTest + "nb_correct\":" + nCorrect + "," +
                "\"param_id\":" + paramId + "," +
                ss;
//        String s = "\"" + trainTest + "duration\":" + elapsedTimeNanoSeconds + "," +
//                "\"" + trainTest + "acc\":" + accuracy + "," +
//                "\"" + trainTest + "nb_correct\":" + nCorrect + "," +
//                "\"param_id\":" + paramId + "," +
//                ss +
//                "\"" + trainTest + "predictions\":" + preds + "," +
//                "\"" + trainTest + "proba\":" + probas;
//        if (cvParamsStr != null) s = s + ",\"cv_params\":" + cvParamsStr;
//        if (cvAccStr != null) s = s + ",\"cv_accs\":" + cvAccStr;
        return "{" + s + "}";
    }

}
