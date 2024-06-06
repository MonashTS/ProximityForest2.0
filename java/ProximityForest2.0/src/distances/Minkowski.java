package distances;

import utils.Tools;

public class Minkowski {
    public static double distance(double[] lines, double[] cols, double p) {
        final int n = lines.length;
        final int m = cols.length;
        final int minLen = Math.min(n, m);

        double dist = 0;
        for (int i = 0; i < minLen; i++) {
            dist += Tools.cost(lines[i], cols[i], p);
        }
//        return Math.pow(dist, 1 / p);
        return dist;
    }

    public static double distance(double[] lines, double[] cols, double cutoff, double p) {
        final int n = lines.length;
        final int m = cols.length;
        final int minLen = Math.min(n, m);

        double dist = 0;
        for (int i = 0; i < minLen; i++) {
            dist += Tools.cost(lines[i], cols[i], p);
            if (dist >= cutoff)
                return Double.POSITIVE_INFINITY;
        }
//        return Math.pow(dist, 1 / p);
        return dist;
    }

}
