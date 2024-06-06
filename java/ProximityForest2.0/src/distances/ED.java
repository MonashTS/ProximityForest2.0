package distances;

import utils.Tools;

public class ED {

    public static double distance(double[] lines, double[] cols) {
        final int n = lines.length;
        final int m = cols.length;
        final int minLen = Math.min(n, m);

        double dist = 0;
        for (int i = 0; i < minLen; i++) {
            final double diff = lines[i] - cols[i];
            dist += diff * diff;
        }
        return dist; // actually squared euclidean distance
    }

    public static double distance(double[] lines, double[] cols, double cutoff) {
        final int n = lines.length;
        final int m = cols.length;
        final int minLen = Math.min(n, m);

        double dist = 0;
        for (int i = 0; i < minLen; i++) {
            final double diff = lines[i] - cols[i];
            dist += diff * diff;
            if (dist >= cutoff)
                return Double.POSITIVE_INFINITY;
        }
        return dist; // actually squared euclidean distance
    }

    public static double distanceGe(double[] lines, double[] cols, double ge) {
        final int n = lines.length;
        final int m = cols.length;
        final int minLen = Math.min(n, m);

        double dist = 0;
        for (int i = 0; i < minLen; i++) {
            dist += Tools.cost(lines[i], cols[i], ge);
        }
        return dist;
    }

    public static double distanceGe(double[] lines, double[] cols, double cutoff, double ge) {
        final int n = lines.length;
        final int m = cols.length;
        final int minLen = Math.min(n, m);

        double dist = 0;
        for (int i = 0; i < minLen; i++) {
            dist += Tools.cost(lines[i], cols[i], ge);
            if (dist >= cutoff)
                return Double.POSITIVE_INFINITY;
        }
        return dist; // actually squared euclidean distance
    }

}
