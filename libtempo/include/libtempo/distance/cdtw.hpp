#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

namespace libtempo::distance {

  namespace internal {

    /** Constrained Dynamic Time Warping with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam FDist        Distance computation function, must be a (size_t, size_t)->FloatType
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist          Distance function, has to capture the series as it only gets the (li,co) coordinate
     * @param w             Half-window parameter (looking at w cells on each side of the diagonal)
     *                      Must be 0<=w<=nblines and nblines - nbcols <= w
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return CDTW between the two series or +INF if early abandoned or, given w, no alignment is possible.
     */
    template<typename FloatType, typename FDist>
    [[nodiscard]] inline FloatType cdtw(size_t nblines, size_t nbcols, FDist dist, const size_t w, FloatType cutoff) {
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

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const FloatType ub = nextafter(cutoff, PINF)-dist(nblines-1, nbcols-1);

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
      // Initialisation of the top border: already initialized to +INF. Initialise the left corner to 0.
      buffers[c-1] = 0;

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
          buffers[c+jStart-1] = cost;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start && j<prev_pp; ++j) {
          const auto d = dist(i, j);
          cost = std::min(buffers[p+j-1], buffers[p+j])+d;
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          const auto d = dist(i, j);
          cost = min(cost, buffers[p+j-1], buffers[p+j])+d;
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<jStop) { // If so, two cases.
          const auto d = dist(i, j);
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffers[p+j-1]+d;
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines-1 && j==nbcols-1 && cost<=cutoff) { return cost; }
              else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(cost, buffers[p+j-1])+d;
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; }
            else { next_start = nbcols-1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp && j<jStop; ++j) {
          const auto d = dist(i, j);
          cost = cost+d;
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
  * @param w           Half warping window, in absolute size (look 'w' cells on each side of the diagonal)
  * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
  *                    May lead to early abandoning.
  *                    If not provided, defaults to PINF,
  *                    which triggers the computation of an upper bound based on the diagonal.
  *                    Use QNAN to run without any upper bounding.
  * @return DTW between the two series
  */
  template<typename FloatType, typename FDist>
  [[nodiscard]] FloatType
  cdtw(size_t length1, size_t length2, FDist dist, size_t w, FloatType ub = utils::PINF<double>) {
    const auto check_result = check_order_series<FloatType>(length1, length2);
    switch (check_result.index()) {
      case 0: { return std::get<0>(check_result); }
      case 1: {
        const auto[nblines, nbcols] = std::get<1>(check_result);
        // Cap the windows and check that, given the constraint, an alignment is possible
        if (w > nblines) { w = nblines; }
        if (nblines - nbcols > w) { return utils::PINF<FloatType>; }
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
        return internal::cdtw<FloatType>(nblines, nbcols, dist, w, ub);
      }
      default: utils::should_not_happen();
    }
  }


  /// Helper with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType cdtw(const auto& s1, const auto& s2, auto mkdist, size_t w, FloatType ub) {
    return cdtw(s1.size(), s2.size(), mkdist(s1, s2), w, ub);
  }

  /// Helper with the sqed as the default distance builder, and a default ub at +INF
  template<typename FloatType>
  [[nodiscard]] inline FloatType cdtw(const auto& s1, const auto& s2, size_t w, FloatType ub = utils::PINF<FloatType>) {
    return cdtw(s1.size(), s2.size(), distance::sqed<FloatType>(s1, s2), w, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType cdtw(const auto& s1, const auto& s2, size_t ndim, auto mkdist, size_t w, FloatType ub) {
    return cdtw(s1.size()/ndim, s2.size()/ndim, mkdist(s1, s2, ndim), w, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType cdtw(const auto& s1, const auto& s2, size_t ndim, size_t w, FloatType ub = utils::PINF<FloatType>) {
    return cdtw(s1.size()/ndim, s2.size()/ndim, distance::sqed<FloatType>(s1, s2, ndim), w, ub);
  }






} // End of namespace libtempo::distance
