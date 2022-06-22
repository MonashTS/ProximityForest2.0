#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/distance/cost_function.hpp>

namespace tempo::distance {

  namespace internal {

    /** Warning
     * In the following, nblines and nbcols are the length of the series.
     * If multivariate series are encoded e.g. in a vector<double>,
     * this is size of the vector divided by the number of dimensions (and not just the size of the vector).
     */

    /** Unconstrained (no window) Dynamic Time Warping, with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(L) space.
     *  Worst case scenario has a O(L²) time complexity (no pruning nor early abandoning).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist          Distance function of type FDist
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @param buffer_v      The buffer used to carry the computation.
     * @return DTW between the two series or +INF if early abandoned.
     *
     * Note: nblines and nbcols are the length of the series - if multivariate series are encoded in a vector<double>,
     * this is not the size of the vector, but the size of the vector divided by the number of dimensions.
     */
    [[nodiscard]] inline F dtw_w_full(const size_t nblines,
                                      const size_t nbcols,
                                      CFun auto dist,
                                      const F cutoff,
                                      std::vector<F> buffer_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      // Adapt constants to the floating point type
      using utils::PINF;
      using utils::min;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = nextafter(cutoff, PINF) - dist(nblines - 1, nbcols - 1);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      buffer_v.assign(nbcols*2, 0);
      auto *buffer = buffer_v.data();
      size_t c{0}, p{nbcols};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      F cost;

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the first line.
      {
        // Fist cell is a special case.
        // Check against the original upper bound dealing with the case where we have both series of length 1.
        cost = dist(0, 0);
        if (cost>cutoff) { return PINF; }
        buffer[c + 0] = cost;
        // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
        // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
        size_t curr_pp = 1;
        for (j = 1; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + dist(0, j);
          buffer[c + j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        ++i;
        prev_pp = curr_pp;
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffer[p + j] + dist(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          cost = std::min(buffer[p + j - 1], buffer[p + j]) + dist(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(cost, buffer[p + j - 1], buffer[p + j]) + dist(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffer[p + j - 1] + dist(i, j);
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; } else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(cost, buffer[p + j - 1]) + dist(i, j);
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; } else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + dist(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        // --- --- ---
        prev_pp = curr_pp;
      } // End of main loop for(;i<nblines;++i)

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols&&cost<=cutoff) { return cost; } else { return PINF; }
    }

    /** Constrained Dynamic Time Warping with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(L) space.
     *  Worst case scenario has a O(L*w) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist          Distance function of type FDist
     * @param w             Warping window
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @param buffer_v      The buffer used to carry the computation.
     * @return DTW between the two series or +INF if early abandoned.
     */
    [[nodiscard]] inline F dtw(const size_t nblines,
                               const size_t nbcols,
                               CFun auto dist,
                               const size_t w,
                               const F cutoff,
                               std::vector<F>& buffer_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      // Adapt constants to the floating point type
      using namespace utils;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = nextafter(cutoff, PINF) - dist(nblines - 1, nbcols - 1);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, init to +INF.
      // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
      buffer_v.assign((1 + nbcols)*2, PINF);
      auto *buffer = buffer_v.data();
      size_t c{0 + 1}, p{nbcols + 2};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      double cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the top border: already initialized to +INF. Initialise the left corner to 0.
      buffer[c - 1] = 0;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        const size_t jStart = std::max(cap_start_index_to_window(i, w), next_start);
        const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
        next_start = jStart;
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Initialise the left border
        {
          cost = PINF;
          buffer[c + jStart - 1] = cost;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          const auto d = dist(i, j);
          cost = std::min(buffer[p + j - 1], buffer[p + j]) + d;
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          const auto d = dist(i, j);
          cost = min(cost, buffer[p + j - 1], buffer[p + j]) + d;
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<jStop) { // If so, two cases.
          const auto d = dist(i, j);
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffer[p + j - 1] + d;
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; }
              else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(cost, buffer[p + j - 1]) + d;
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; }
            else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<jStop; ++j) {
          const auto d = dist(i, j);
          cost = cost + d;
          buffer[c + j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        // --- --- ---
        prev_pp = curr_pp;
      } // End of main loop for(;i<nblines;++i)

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols&&cost<=cutoff) { return cost; } else { return PINF; }
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // --- END OF INTERNAL
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  } // End of namespace internal

  /** Early Abandoned and Pruned Dynamic Time Warping.
   * If the warping window is w =< L-2, use the constrained implementation; else use the unconstrained one.
   * Having w > L-2 is the same as having no window.
   * @param length1     Length of the first series.
   * @param length2     Length of the second series.
   * @param dist        Distance function of type FDist
   * @param w           Half warping window, in absolute size (look 'w' cells on each side of the diagonal)
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    ub = PINF: use pruning
   *                    ub = QNAN: no lower bounding
   *                    ub = other value: use for pruning and early abandoning
   * @return CDTW between the two series or +INF if early abandoned or maximum error (incompatible window).
   */

  [[nodiscard]] F dtw(size_t nblines,
                      size_t nbcols,
                      CFun auto dist,
                      size_t w,
                      F ub,
                      std::vector<F>& buffer_v
  ) {
    using utils::PINF;
    if (nblines==0&&nbcols==0) { return 0; }
    else if ((nblines==0)!=(nbcols==0)) { return PINF; }
    else {
      // Check that the window allows for an alignment
      // If this is accepted, we do not need to check the window when computing a new UB
      const auto m = std::min(nblines, nbcols);
      const auto M = std::max(nblines, nbcols);
      if (M - m>w) { return PINF; }
      // Compute a cutoff point using the diagonal
      if (std::isinf(ub)) {
        ub = 0;
        // Cover diagonal
        for (size_t i{0}; i<m; ++i) { ub = ub + dist(i, i); }
        // Fewer line than columns: complete the last line
        if (nblines<nbcols) { for (size_t i{nblines}; i<nbcols; ++i) { ub = ub + dist(nblines - 1, i); }}
          // Fewer columns than lines: complete the last column
        else if (nbcols<nblines) { for (size_t i{nbcols}; i<nblines; ++i) { ub = ub + dist(i, nbcols - 1); }}
      } else if (std::isnan(ub)) { ub = PINF; }
      // ub computed: choose the version to call
      if (w>M - 2) {
        return internal::dtw_w_full(nblines, nbcols, dist, ub, buffer_v);
      } else {
        return internal::dtw(nblines, nbcols, dist, w, ub, buffer_v);
      }
    }
  }

  /// Helper without having to provide a buffer
  [[nodiscard]] inline F dtw(size_t nblines, size_t nbcols, CFun auto dist, size_t w, F ub) {
    std::vector<F> v;
    return dtw(nblines, nbcols, dist, w, ub, v);
  }

  /// Helper for TSLike, without having to provide a buffer
  template<TSLike T>
  [[nodiscard]] inline F
  dtw(const T& lines, const T& cols, CFunBuilder<T> auto mkdist, size_t w, F ub) {
    const auto ls = lines.length();
    const auto cs = cols.length();
    const CFun auto dist = mkdist(lines, cols);
    std::vector<F> v;
    return dtw(ls, cs, dist, w, ub, v);
  }

} // End of namespace tempo::distance
