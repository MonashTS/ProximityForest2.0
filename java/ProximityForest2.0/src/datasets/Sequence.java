package datasets;


import java.io.Serializable;
import java.util.HashMap;

import static transforms.Transforms.TimeSeriesTransforms;
import static transforms.Transforms.TimeSeriesTransforms.raw;

public class Sequence implements Serializable {
    private static final long serialVersionUID = 1L;

    public double[][] data; // data in multivariate format, dim x length
    public int classLabel;
    public boolean isNorm;
    public TimeSeriesTransforms type = raw;
    public HashMap<TimeSeriesTransforms, double[]> transforms;

    public Sequence(final double[][] series, final int label) {
        this.data = series;
        this.classLabel = label;
        this.isNorm = false;
    }

    public Sequence(final double[] series, final int label) {
        this.data = new double[1][series.length];
        this.data[0] = series;
        this.classLabel = label;
        this.isNorm = false;
    }

    public void setLabel(final int y) {
        this.classLabel = y;
    }

    public double[][] get() {
        return data;
    }

    public double[] get(final int channel) {
        return data[channel];
    }

    public double get(final int channel, final int index) {
        return data[channel][index];
    }

    public double[] firstChannel() {
        return data[0];
    }

    public int length() {
        return data[0].length;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.classLabel);
        sb.append(":");

        int max = 10;
        int i, j;
        for (i = 0; i < Math.min(data.length, max); i++) {
            for (j = 0; j < Math.min(data[i].length, max); j++) {
                sb.append(data[i][j]);
                sb.append(",");
            }
            if (j == max) {
                sb.append("...");
                sb.append(";");
            }
        }
        if (i == max) {
            sb.append("...");
        }


        return sb.toString();
    }

}

