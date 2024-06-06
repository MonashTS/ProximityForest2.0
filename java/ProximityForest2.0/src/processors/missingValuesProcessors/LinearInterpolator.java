package processors.missingValuesProcessors;


import utils.Tools;

public class LinearInterpolator extends MissingValuesProcessor {
    @Override
    public double[] process(double[] tmp) {
        int prev;
        int next;
        int j = 0;
        while (j < tmp.length) {
            if (Double.isNaN(tmp[j])) {
                prev = j - 1;
                next = j + 1;
                for (int k = j - 1; k >= 0; k--)
                    if (!Tools.isMissing(tmp[k])) {
                        prev = k; // the index that is not NaN
                        break;
                    }

                boolean nextFound = false;
                for (int k = j + 1; k < tmp.length; k++) {
                    if (!Tools.isMissing(tmp[k])) {
                        next = k; // the index that is not NaN
                        nextFound = true;
                        break;
                    }
                }
                if (!nextFound) {
                    next = tmp.length;
                }

                if (prev < 0) {
                    for (int k = 0; k < next; k++) {
                        tmp[k] = tmp[next];
                    }
                } else if (next >= tmp.length) {
                    for (int k = prev + 1; k < tmp.length; k++) {
                        tmp[k] = tmp[prev];
                    }
                } else {
                    for (int k = prev + 1; k < next; k++) {
                        tmp[k] = Tools.linearInterp(k, tmp[prev], prev, tmp[next], next);
                    }
                }
                j = next;
            }
            j++;
        }
        return tmp;
    }
}
