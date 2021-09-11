#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

namespace libtempo::distance {

  namespace internal {

    /** Edit Distance with Real Penalty (ERP), with cut-off point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @tparam FDist        Distance computation function, must be a (const D&, size_t, constD&, size_t)->FloatType
     * @param lines         Data for the lines
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param cols          Data for the lines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist          Distance function of type FDist
     * @param gValue        Data for the gValue - must have the correct dimension!
     * @param w             Half-window parameter (looking at w cells on each side of the diagonal)
     *                      Must be 0<=w<=nblines and nblines - nbcols <= w
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return ERP value or +INF if early abandoned, or , given w, no alignment is possible
     */
    template<typename FloatType, typename D, typename FDist>
    [[nodiscard]] inline FloatType erp(
      const D& lines, size_t nblines,
      const D& cols, size_t nbcols,
      FDist dist,
      const D& gValue, size_t w,
      FloatType cutoff
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      assert(nbcols<=nblines);
      assert(w<=nblines);
      assert(nblines-nbcols<=w);
      // Adapt constants to the floating point type
      using namespace utils;
      constexpr auto PINF = utils::PINF<FloatType>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const FloatType ub = utils::initBlock {
        const auto la = min(
          dist(gValue, 0, cols, nbcols-1),          // Previous
          dist(lines, nblines-1, cols, nbcols-1),   // Diagonal
          dist(lines, nblines-1, gValue, 0)         // Above
        );
        return nextafter(cutoff, PINF)-la;
      };

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, init to +INF.
      // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
      std::vector<FloatType> buffers_v((1+nbcols)*2, PINF);
      auto* buffers = buffers_v.data();
      size_t c{0+1}, p{nbcols+2};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      double cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the top border
      {   // Matrix Border - Top diagonal
        buffers[c-1] = 0;
        // Matrix Border - First line
        const size_t jStop = cap_stop_index_to_window_or_end(0, w, nbcols);
        for (j = 0; buffers[c+j-1]<=ub && j<jStop; ++j) {
          buffers[c+j] = buffers[c+j-1]+dist(gValue, 0, cols, j);
        }
        // Pruning point set to first +INF value (or out of bound)
        prev_pp = j;
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Part 1: Loop with computed left border.
      {   // The left border has a computed value while it's within the window and its value bv <= ub
        // No "front pruning" (next_start) and no early abandoning can occur while in this loop.
        const size_t iStop = cap_stop_index_to_window_or_end(0, w, nblines);
        for (; i<iStop; ++i) {
          // --- --- --- Variables init
          constexpr size_t jStart = 0;
          const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
          j = jStart;
          size_t curr_pp = jStart; // Next pruning point init at the start of the line
          // --- --- --- Stage 0: Initialise the left border
          {
            // We haven't swap yet, so the 'top' cell is still indexed by 'c-1'.
            cost = buffers[c-1]+dist(lines, i, gValue, 0);
            if (cost>ub) { break; }
            else {
              std::swap(c, p);
              buffers[c-1] = cost;
            }
          }
          // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
          // No stage 1 here.
          // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
          for (; j<prev_pp; ++j) {
            cost = min(
              cost+dist(gValue, 0, cols, j),          // Previous
              buffers[p+j-1]+dist(lines, i, cols, j), // Diagonal
              buffers[p+j]+dist(lines, i, gValue, 0)  // Above
            );
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
          }
          // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
          if (j<jStop) { // Possible path in previous cells: left and diag.
            cost = std::min(
              cost+dist(gValue, 0, cols, j),           // Previous
              buffers[p+j-1]+dist(lines, i, cols, j)   // Diagonal
            );
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
            ++j;
          }
          // --- --- --- Stage 4: After the previous pruning point: only prev.
          // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
          for (; j==curr_pp && j<jStop; ++j) {
            cost = cost+dist(gValue, 0, cols, j);  // Previous
            buffers[c+j] = cost;
            if (cost<=ub) { ++curr_pp; }
          }
          // --- --- ---
          prev_pp = curr_pp;
        }
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Part 2: Loop with +INF left border
      {
        for (; i<nblines; ++i) {
          // --- --- --- Swap and variables init
          std::swap(c, p);
          const size_t jStart = std::max(cap_start_index_to_window(i, w), next_start);
          const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
          j = jStart;
          next_start = jStart;
          size_t curr_pp = jStart; // Next pruning point init at the start of the line
          // --- --- --- Stage 0: Initialise the left border
          {
            cost = PINF;
            buffers[c+jStart-1] = cost;
          }
          // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
          for (; j==next_start && j<prev_pp; ++j) {
            cost = std::min(
              buffers[p+j-1]+dist(lines, i, cols, j),  // Diagonal
              buffers[p+j]+dist(lines, i, gValue, 0)   // Above
            );
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
          }
          // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
          for (; j<prev_pp; ++j) {
            cost = min(
              cost+dist(gValue, 0, cols, j),           // Previous
              buffers[p+j-1]+dist(lines, i, cols, j),  // Diagonal
              buffers[p+j]+dist(lines, i, gValue, 0)   // Above
            );
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
          }
          // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
          if (j<jStop) { // If so, two cases.
            if (j==next_start) { // Case 1: Advancing next start: only diag.
              cost = buffers[p+j-1]+dist(lines, i, cols, j);     // Diagonal
              buffers[c+j] = cost;
              if (cost<=ub) { curr_pp = j+1; }
              else {
                // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                if (i==nblines-1 && j==nbcols-1 && cost<=cutoff) { return cost; } else { return PINF; }
              }
            } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
              cost = std::min(
                cost+dist(gValue, 0, cols, j),           // Previous
                buffers[p+j-1]+dist(lines, i, cols, j)   // Diagonal
              );
              buffers[c+j] = cost;
              if (cost<=ub) { curr_pp = j+1; }
            }
            ++j;
          } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
            if (j==next_start) {
              // But only if we are above the original UB
              // Else set the next starting point to the last valid column
              if (cost>cutoff) { return PINF; } else { next_start = nbcols-1; }
            }
          }
          // --- --- --- Stage 4: After the previous pruning point: only prev.
          // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
          for (; j==curr_pp && j<jStop; ++j) {
            cost = cost+dist(gValue, 0, cols, j);
            buffers[c+j] = cost;
            if (cost<=ub) { ++curr_pp; }
          }
          // --- --- ---
          prev_pp = curr_pp;
        }
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols && cost<=cutoff) { return cost; } else { return PINF; }
    }

  } // End of namespace internal

  /** Edit Distance with Real Penalty (ERP), with cut-off point for early abandoning and pruning.
   *  Double buffered implementation using O(n) space.
   *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
   *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
   *  Actual implementation assuming that some pre-conditions are fulfilled.
   * @tparam FloatType  The floating number type used to represent the series.
   * @tparam D          Type of underlying collection - given to dist
   * @tparam FDist      Distance computation function, must be a (const D&, size_t, constD&, size_t)->FloatType
   * @param series1     Data for the first series
   * @param length1     Length of the first series.
   * @param series2     Data for the second series
   * @param length2     Length of the second series.
   * @param dist        Distance function of type FDist
   * @param gv          Data for the gValue - must have the correct dimension!
   * @param w           Half-window parameter (looking at w cells on each side of the diagonal)
   *                    Must be 0<=w<=nblines and nblines - nbcols <= w
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    ub = PINF: use pruning
   *                    ub = QNAN: no lower bounding
   *                    ub = other value: use for pruning and early abandoning
   * @return ERP value or +INF if early abandoned, or , given w, no alignment is possible
   */
  template<typename FloatType, typename D, typename FDist>
  [[nodiscard]] FloatType erp(
    const D& series1, size_t length1,
    const D& series2, size_t length2,
    FDist dist,
    const D& gv, size_t w,
    FloatType ub = utils::PINF<double>
  ) {
    const auto check_result = check_order_series<FloatType>(series1, length1, series2, length2);
    switch (check_result.index()) {
      case 0: { return std::get<0>(check_result); }
      case 1: {
        const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
        // Cap the windows and check that, given the constraint, an alignment is possible
        if (w>nblines) { w = nblines; }
        if (nblines-nbcols>w) { return utils::PINF<FloatType>; }
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        if (std::isinf(ub)) {
          ub = 0;
          // We know that nbcols =< nblines: cover all the columns, then cover the remaining line in the last column
          for (size_t i{0}; i<nbcols; ++i) { ub += dist(lines, i, cols, i); }
          for (size_t i{nbcols}; i < nblines; ++i) { ub += dist(lines, i, gv, 0); }
        } else if (std::isnan(ub)) {
          ub = utils::PINF<FloatType>;
        }
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::erp<FloatType>(lines, nblines, cols, nbcols, dist, gv, w, ub);
      }
      default: utils::should_not_happen();
    }
  }

  /// Helper with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType erp(
    const D& s1, const D& s2, auto mkdist,
    FloatType gv, size_t w, FloatType ub
  ) {
    return erp(s1, s1.size(), s2, s2.size(), mkdist(), {gv}, w, ub);
  }

  /// Helper with the sqed as the default distance builder, and a default ub at +INF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType erp(
    const D& s1, const D& s2,
    FloatType gv, size_t w,
    FloatType ub = utils::PINF<FloatType>
  ) {
    return erp(s1, s1.size(), s2, s2.size(), distance::sqed<FloatType, D>(), {gv}, w, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType erp(
    const D& s1, const D& s2, size_t ndim, auto mkdist,
    const D& gv, size_t w,
    FloatType ub
  ) {
    return erp(s1, s1.size()/ndim, s2, s2.size()/ndim, mkdist(ndim), gv, w, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType erp(
    const D& s1, const D& s2, size_t ndim,
    const D& gv, size_t w,
    FloatType ub = utils::PINF<FloatType>
  ) {
    return erp(s1, s1.size()/ndim, s2, s2.size()/ndim, distance::sqed<FloatType, D>(ndim), gv, w, ub);
  }


} // End of namespace libtempo::distance
