#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

namespace libtempo::distance {

  namespace internal {

    /** Amerced Dynamic Time Warping, with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @tparam FDist        Distance computation function, must be a (const D&, size_t, constD&, size_t)->FloatType
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param gamma         Cost function of type FDist
     * @param penalty       Fixed cost penalty for warping steps; must be >=0
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return CDTW between the two series or +PINF if early abandoned or, given w, no alignment is possible.
     */
    template<typename FloatType, typename D, typename FDist>
    [[nodiscard]] inline FloatType adtw(
      const D& lines, size_t nblines,
      const D& cols, size_t nbcols,
      FDist gamma,
      const FloatType penalty,
      FloatType cutoff
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      assert(nbcols<=nblines);
      // Adapt constants to the floating point type
      constexpr auto PINF = utils::PINF<FloatType>;
      using utils::min;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly uadtw in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const double ub = nextafter(cutoff, PINF)-gamma(lines, nblines-1, cols, nbcols-1);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      auto buffers = std::unique_ptr<double[]>(new double[nbcols*2]);
      size_t c{0}, p{nbcols};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also uadtw as the "left neighbour".
      double cost;

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the first line.
      {
        // Fist cell is a special case.
        // Check against the original upper bound dealing with the case where we have both series of length 1.
        cost = gamma(lines, 0, cols, 0);
        if (cost>cutoff) { return PINF; }
        buffers[c+0] = cost;
        // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
        // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
        size_t curr_pp = 1;
        for (j = 1; j==curr_pp && j<nbcols; ++j) {
          cost = cost+gamma(lines, 0, cols, j)+penalty; // Left: penalty
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
          cost = buffers[p+j]+gamma(lines, i, cols, j)+penalty; // Top: penalty
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start && j<prev_pp; ++j) {
          const auto d = gamma(lines, i, cols, j);
          cost = std::min(
            buffers[p+j-1]+d,         // Diag: no penalty
            buffers[p+j]+d+penalty     // Top: penalty
          );
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          const auto d = gamma(lines, i, cols, j);
          cost = min(d+cost+penalty,   // Left: penalty
            buffers[p+j-1]+d,         // Diag: no penalty
            buffers[p+j]+d+penalty);   // Top: penalty
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          const auto d = gamma(lines, i, cols, j);
          if (j==next_start) { // Case 1: Advancing next start: only diag (no penalty)
            cost = buffers[p+j-1]+d;
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines-1 && j==nbcols-1 && cost<=cutoff) { return cost; }
              else {
                return PINF;
              }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left (penalty) and diag.
            cost = std::min(cost+d+penalty, buffers[p+j-1]+d);
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost>cutoff) {
              return PINF;
            }
            else { next_start = nbcols-1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp && j<nbcols; ++j) {
          const auto d = gamma(lines, i, cols, j);
          cost = cost+d+penalty; // Left: penalty
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
   * @tparam D          Type of underlying collection - given to dist
   * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
   * @param series1     Data for the first series
   * @param length1     Length of the first series.
   * @param series2     Data for the second series
   * @param length2     Length of the second series.
   * @param dist        Cost function of type FDist
   * @param penalty     Additive penalty for warping steps
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    ub = PPINF: use pruning
   *                    ub = QNAN: no lower bounding
   *                    ub = other value: use for pruning and early abandoning
   * @return CDTW between the two series or +PINF if early abandoned or maximum error (incompatible window).
   */
  template<typename FloatType, typename D, typename FDist>
  [[nodiscard]] FloatType
  adtw(const D& series1, size_t length1, const D& series2, size_t length2, FDist dist, const FloatType penalty,
    FloatType ub = utils::PINF<double>) {
    const auto check_result = check_order_series<FloatType>(series1, length1, series2, length2);
    switch (check_result.index()) {
      case 0: { return std::get<0>(check_result); }
      case 1: {
        const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        if (std::isinf(ub)) {
          ub = 0;
          // We know that nbcols =< nblines: cover all the columns, then cover the remaining line in the last column
          // Last column must add the penalty (warping step)
          // Warning: the order of operations does matter in floating point operations; do not use 'ub += ...'
          for (size_t i{0}; i<nbcols; ++i) { ub = ub + dist(lines, i, cols, i); }
          for (size_t i{nbcols}; i<nblines; ++i) { ub = ub + dist(lines, i, cols, nbcols-1)+penalty; }
        } else if (std::isnan(ub)) {
          ub = utils::PINF<FloatType>;
        }
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::adtw<FloatType>(lines, nblines, cols, nbcols, dist, penalty, ub);
      }
      default: utils::should_not_happen();
    }
  }

  /// Helper with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType adtw(const auto& s1, const auto& s2, auto mkdist, FloatType penalty, FloatType ub) {
    return adtw(s1, s1.size(), s2, s2.size(), mkdist(), penalty, ub);
  }

  /// Helper with the sqed as the default distance builder, and a default ub at +PINF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType adtw(const D& s1, const D& s2, FloatType penalty, FloatType ub = utils::PINF<FloatType>) {
    return adtw(s1, s1.size(), s2, s2.size(), distance::sqed<FloatType, D>(), penalty, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType adtw(const D& s1, const D& s2, size_t ndim, auto mkdist, FloatType penalty, FloatType ub) {
    return adtw(s1, s1.size()/ndim, s2, s2.size()/ndim, mkdist(ndim), penalty, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType
  adtw(const D& s1, const D& s2, size_t ndim, FloatType penalty, FloatType ub = utils::PINF<FloatType>) {
    return adtw(s1, s1.size()/ndim, s2, s2.size()/ndim, distance::sqed<FloatType, D>(ndim), penalty, ub);
  }


} // End of namespace libtempo::distance