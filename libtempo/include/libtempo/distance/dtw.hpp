#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

namespace libtempo::distance {

  namespace internal {

    /** Dynamic Time Warping with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType  The floating number type used to represent the series.
     * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
     * @param length1     Length of the first series.
     * @param length2     Length of the second series.
     * @param dist        Distance function, has to capture the series as it only gets the (li,co) coordinate
     * @param cutoff.     Attempt to prune computation of alignments with cost > cutoff.
     *                    May lead to early abandoning.
     * @return DTW between the two series or +INF if early abandoned.
     */
    template<typename FloatType, typename FDist>
    [[nodiscard]] inline FloatType dtw(size_t nblines, size_t nbcols, FDist dist, const FloatType cutoff) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      assert(nbcols<=nblines);
      // Adapt constants to the floating point type
      constexpr auto PINF = utils::PINF<FloatType>;
      using utils::min;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const FloatType ub = nextafter(cutoff, PINF)-dist(nblines-1, nbcols-1);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      auto buffers = std::unique_ptr<FloatType[]>(new FloatType[nbcols*2]);
      size_t c{0}, p{nbcols};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      FloatType cost;

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
        buffers[c+0] = cost;
        // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
        // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
        size_t curr_pp = 1;
        for (j = 1; j==curr_pp && j<nbcols; ++j) {
          cost = cost+dist(0, j);
          buffers[c+j] = cost;
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
          cost = buffers[p+j]+dist(i, j);
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start && j<prev_pp; ++j) {
          cost = std::min(buffers[p+j-1], buffers[p+j])+dist(i, j);
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(cost, buffers[p+j-1], buffers[p+j])+dist(i, j);
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffers[p+j-1]+dist(i, j);
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines-1 && j==nbcols-1 && cost<=cutoff) { return cost; } else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(cost, buffers[p+j-1])+dist(i, j);
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
        for (; j==curr_pp && j<nbcols; ++j) {
          cost = cost+dist(i, j);
          buffers[c+j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        // --- --- ---
        prev_pp = curr_pp;
      } // End of main loop for(;i<nblines;++i)

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols && cost<=cutoff) { return cost; } else { return PINF; }
    }
  } // End of namespace internal




  /** Early Abandoned and Pruned Dynamic Time Warping.
   * @tparam FloatType  The floating number type used to represent the series.
   * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
   * @param length1     Length of the first series.
   * @param length2     Length of the second series.
   * @param dist        Distance function, has to capture the series as it only gets the (li,co) coordinates
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    ub = PINF: use pruning
   *                    ub = QNAN: no lower bounding
   *                    ub = other value: use for pruning and early abandoning
   * @return DTW between the two series
  */
  template<typename FloatType, typename FDist>
  [[nodiscard]] FloatType dtw(size_t length1, size_t length2, FDist dist, FloatType ub) {
    const auto check_result = check_order_series<FloatType>(length1, length2);
    switch (check_result.index()) {
      case 0: { return std::get<0>(check_result); }
      case 1: {
        const auto[nblines, nbcols] = std::get<1>(check_result);
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        if (ub==utils::PINF<FloatType>) {
          ub = 0;
          // We know that nbcols =< nblines: cover all the columns, then cover the remaining line in the last column
          for (size_t i{0}; i<nbcols; ++i) { ub += dist(i, i); }
          for (size_t i{nbcols}; i<nblines; ++i) { ub += dist(i, nbcols-1); }
        } else if (std::isnan(ub)) {
          ub = utils::PINF<FloatType>;
        }
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::dtw<FloatType>(nblines, nbcols, dist, ub);
      }
      default: utils::should_not_happen();
    }
  }

  /// Helper with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType dtw(const auto& s1, const auto& s2, auto mkdist, FloatType ub) {
    return dtw(s1.size(), s2.size(), mkdist(s1, s2), ub);
  }

  /// Helper with the sqed as the default distance builder, and a default ub at +INF
  template<typename FloatType>
  [[nodiscard]] inline FloatType dtw(const auto& s1, const auto& s2, FloatType ub = utils::PINF<FloatType>) {
    return dtw(s1.size(), s2.size(), distance::sqed<FloatType>(s1, s2), ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType dtw(const auto& s1, const auto& s2, size_t ndim, auto mkdist, FloatType ub) {
    return dtw(s1.size()/ndim, s2.size()/ndim, mkdist(s1, s2, ndim), ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType dtw(const auto& s1, const auto& s2, size_t ndim, FloatType ub = utils::PINF<FloatType>) {
    return dtw(s1.size()/ndim, s2.size()/ndim, distance::sqed<FloatType>(s1, s2, ndim), ub);
  }


} // End of namespace libtempo::distance
