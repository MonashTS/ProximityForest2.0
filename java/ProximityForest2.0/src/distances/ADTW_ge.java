package distances;

import utils.Tools;

import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.min;
import static utils.Tools.EPSILON;

public class ADTW_ge extends ADTW {
    public static double distance(double[] lines, double[] cols, double weight, double cutoff, double ge) {
        // Ensure that lines are longer than columns
        if (lines.length < cols.length) {
            double[] swap = lines;
            lines = cols;
            cols = swap;
        }

        // --- --- --- Declarations
        int nblines = lines.length;
        int nbcols = cols.length;

        // Setup buffers - no extra initialization required - border condition managed in the code.
        double[] buffers = new double[2 * nbcols];
        int c = 0;
        int p = nbcols;

        // Line & columns indices
        int i = 0;
        int j = 0;

        // Cost accumulator in a line, also used as the "left neighbor"
        double cost = 0;

        // EAP variable: track where to start the next line, and the position of the previous pruning point.
        // Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
        int next_start = 0;
        int prev_pp = 0;

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Create a new tighter upper bounds (most commonly uadtw in the code).
        // First, take the "next float" after "cutoff" to deal with numerical instability.
        // Then, subtract the cost of the last alignment.
        double ub = cutoff + EPSILON - Tools.cost(lines[nblines - 1], cols[nbcols - 1], ge);

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Initialisation of the first line.
        {
            double l0 = lines[0];
            // Fist cell is a special case.
            // Check against the original upper bound dealing with the case where we have both series of length 1.
            cost = Tools.cost(l0, cols[0], ge);
            if (cost > cutoff) {
                return POSITIVE_INFINITY;
            }
            buffers[c + 0] = cost;
            // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
            // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
            int curr_pp = 1;
            for (j = 1; j == curr_pp && j < nbcols; ++j) {
                cost = cost + Tools.cost(l0, cols[j], ge) + weight; // Left: penalty
                buffers[c + j] = cost;
                if (cost <= ub) {
                    ++curr_pp;
                }
            }
            ++i;
            prev_pp = curr_pp;
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Main loop
        for (; i < nblines; ++i) {
            // --- --- --- Swap and variables init
            int swap = c;
            c = p;
            p = swap;

            double li = lines[i];
            int curr_pp = next_start; // Next pruning point init at the start of the line
            j = next_start;
            // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
            {
                cost = buffers[p + j] + Tools.cost(li, cols[j], ge) + weight; // Top: penalty
                buffers[c + j] = cost;
                if (cost <= ub) {
                    curr_pp = j + 1;
                } else {
                    ++next_start;
                }
                ++j;
            }
            // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
            for (; j == next_start && j < prev_pp; ++j) {
                final double d = Tools.cost(li, cols[j], ge);
                cost = min(
                        d + buffers[p + j - 1],         // Diag: no penalty
                        d + buffers[p + j] + weight     // Top: penalty
                );
                buffers[c + j] = cost;
                if (cost <= ub) {
                    curr_pp = j + 1;
                } else {
                    ++next_start;
                }
            }
            // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
            for (; j < prev_pp; ++j) {
                final double d = Tools.cost(li, cols[j], ge);
                cost = Tools.min(
                        d + cost + weight,// Left: penalty
                        d + buffers[p + j - 1], // Diag: no penalty
                        d + buffers[p + j] + weight// Top: penalty
                );
                buffers[c + j] = cost;
                if (cost <= ub) {
                    curr_pp = j + 1;
                }
            }
            // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
            if (j < nbcols) { // If so, two cases.
                final double d = Tools.cost(li, cols[j], ge);
                if (j == next_start) { // Case 1: Advancing next start: only diag.
                    cost = buffers[p + j - 1] + d;
                    buffers[c + j] = cost;
                    if (cost <= ub) {
                        curr_pp = j + 1;
                    } else {
                        // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                        if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) {
                            return cost;
                        } else {
                            return POSITIVE_INFINITY;
                        }
                    }
                } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
                    cost = min(d + cost + weight, d + buffers[p + j - 1]);
                    buffers[c + j] = cost;
                    if (cost <= ub) {
                        curr_pp = j + 1;
                    }
                }
                ++j;
            } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
                if (j == next_start) {
                    // But only if we are above the original UB
                    // Else set the next starting point to the last valid column
                    if (cost > cutoff) {
                        return POSITIVE_INFINITY;
                    } else {
                        next_start = nbcols - 1;
                    }
                }
            }
            // --- --- --- Stage 4: After the previous pruning point: only prev.
            // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
            for (; j == curr_pp && j < nbcols; ++j) {
                cost = cost + Tools.cost(li, cols[j], ge) + weight;// Left: penalty
                buffers[c + j] = cost;
                if (cost <= ub) {
                    ++curr_pp;
                }
            }
            // --- --- ---
            prev_pp = curr_pp;
        } // End of main.main loop for(;i<nblines;++i)

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Finalization
        // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
        if (j == nbcols && cost <= cutoff) {
            return cost;
        } else {
            return POSITIVE_INFINITY;
        }
    }
}
