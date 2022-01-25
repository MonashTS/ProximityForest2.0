#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

namespace libtempo::distance {

  namespace internal {

    /** Univariate cost function used when transforming X=(x1, x2, ... xi) into Y = (y1, ..., yj) by Split or Merge (symmetric)
     * @tparam FT    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @param X             Main series: the series where a new point is added (can be line or column!)
     * @param xnew_         In either X or Y
     * @param xi_           Last point of X
     * @param Y             The other series
     * @param yj_           Last point of Y
     * @param c             cost of split and merge operation
     * @return MSM cost of the xi-yj alignment (without "recursive" part)
     */
    template<typename FT, typename D>
    [[nodiscard]] inline FT msm_cost_uni(const D& X, size_t xnew_, size_t xi_, const D& Y, size_t yj_, FT cost) {
      FT xnew = X[xnew_];
      FT xi = X[xi_];
      FT yj = Y[yj_];
      if (((xi<=xnew) && (xnew<=yj)) || ((yj<=xnew) && (xnew<=xi))) { return cost; }
      else { return cost+std::min(std::abs(xnew-xi), std::abs(xnew-yj)); }
    }

    /** Multivariate cost function when transforming
     *  X=(x1, x2, ... xi) into Y = (y1, ..., yj) by Split or Merge (symmetric - dependening how we move in the matrix).
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @tparam FDist        Distance function, must be a (const D&, size_t, constD&, size_t)->FloatType
     * @tparam FDistMP      Distance to midpoint function, must be a (const D&, size_t, size_t, constD&, size_t)->FloatType
     * @param  X            Main series: the series where a new point is added (can be line or column!)
     * @param  xnew         Index of the new point in X
     * @param  Y            The other series
     * @param  xi           Index of the last point in X
     * @param  yi           Index of the last point in Y
     * From Shifaz et al 2021
     * Elastic Similarity Measures for Multivariate Time Series Classification
     * https://arxiv.org/abs/2102.10231
     *   We check if xnew is "between" xi an yi by checking if it is inside the hyper-sphere
     *   defined by xi and yi: take the midpoint between xi and yi and the sphere radius.
     *   If the distance between the midpoint and xnew > radius, xnew is not in the sphere.
     */
    template<typename FloatType, typename D, typename FDist, typename FDistMP>
    [[nodiscard]] inline FloatType msm_cost_multi(
      const D& X, size_t xnew, size_t xi,
      const D& Y, size_t yi,
      FDist dist,
      FDistMP distmpoint,
      FloatType cost
    ) {
      const FloatType radius = dist(X, xi, Y, yi)/2; // distance between xi and yi give us the sphere diameter
      const FloatType d_to_mid = distmpoint(X, xnew, xi, Y, yi);
      if (d_to_mid<=radius) { return cost; }
      else {
        const FloatType d_to_prev = dist(X, xnew, X, xi);
        const FloatType d_to_other = dist(X, xnew, Y, yi);
        return cost+std::min<FloatType>(d_to_prev, d_to_other);
      }
    }

    /** Edit Distance with Real Penalty (MSM), with cut-off point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @tparam FDist        Distance computation function, must be a (const D&, size_t, constD&, size_t)->FloatType
     * @tparam FCost        MSM cost computation function - see msm_cost_uni and msms_cost_multi
     * @param lines         Data for the lines
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param cols          Data for the lines
     * @param dist          Distance function of type FDist
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param msms_cost     MSM cost function
     * @param co            MSM cost parameter, transmitted to msm_cost
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return MSM value or +INF if early abandoned, or , given w, no alignment is possible
     */
    template<typename FloatType, typename D, typename FDist, typename FCost>
    [[nodiscard]] inline FloatType msm(
      const D& lines, size_t nblines,
      const D& cols, size_t nbcols,
      FDist dist,
      FCost msm_cost,
      const FloatType co,
      FloatType cutoff
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      assert(nbcols<=nblines);
      // Adapt constants to the floating point type
      using namespace utils;
      constexpr auto PINF = utils::PINF<FloatType>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const FloatType ub = initBlock {
        // The last alignment can only computed if we have nbcols >= 2
        if (nbcols>=2) {
          const auto i = nblines-1;
          const auto i1 = nblines-2;
          const auto j = nbcols-1;
          const auto j1 = nbcols-2;
          const auto la = min(
            dist(lines, i, cols, j),              // Diag: Move
            msm_cost(cols, j, j1, lines, i, co),  // Previous: Split/Merge
            msm_cost(lines, i, i1, cols, j, co)   // Above: Split/Merge
          );
          return FloatType(nextafter(cutoff, PINF-la));
        } else {
          return FloatType(cutoff); // Force type to prevent auto-deduction failure
        }
      };

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      auto buffers = std::unique_ptr<double[]>(new double[nbcols*2]);
      size_t c{0}, p{nbcols};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      double cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation: compute the first line. Required as the main loop starts at line=1, not 0.
      {
        // First cell (0,0) is a special case. Early abandon if above the cut-off point.
        {
          cost = dist(lines, 0, cols, 0); // Very first cell
          buffers[c+0] = cost;
          if (cost<=ub) { prev_pp = 1; } else { return PINF; }
        }
        // Rest of the line, a cell only depends on the previous cell. Stop when > ub, update prev_pp.
        for (j = 1; j<nbcols; ++j) {
          cost = cost+msm_cost(cols, j, j-1, lines, 0, co);
          if (cost<=ub) {
            buffers[c+j] = cost;
            prev_pp = j+1;
          } else { break; }
        }
        // Next line.
        ++i;
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        const size_t i1 = i-1;
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffers[p+j]+msm_cost(lines, i, i1, cols, j, co);
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start && j<prev_pp; ++j) {
          cost = std::min(
            buffers[p+j-1]+dist(lines, i, cols, j),           // Diag: Move
            buffers[p+j]+msm_cost(lines, i, i1, cols, j, co)  // Above: Split/Merge
          );
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(
            buffers[p+j-1]+dist(lines, i, cols, j),           // Diag: Move
            cost+msm_cost(cols, j, j-1, lines, i, co),        // Previous: Split/Merge
            buffers[p+j]+msm_cost(lines, i, i1, cols, j, co)  // Above: Split/Merge
          );
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffers[p+j-1]+dist(lines, i, cols, j);    // Diag: Move
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines-1 && j==nbcols-1 && cost<=cutoff) { return cost; }
              else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(
              buffers[p+j-1]+dist(lines, i, cols, j),       // Diag: Move
              cost+msm_cost(cols, j, j-1, lines, i, co)     // Previous: Split/Merge
            );
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
        for (; j==curr_pp && j<nbcols; ++j) {
          cost = cost+msm_cost(cols, j, j-1, lines, i, co);    // Previous: Split/Merge
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



  /** Edit Distance with Real Penalty (MSM), with cut-off point for early abandoning and pruning.
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
   * @return MSM value or +INF if early abandoned, or , given w, no alignment is possible
   */
  template<typename FloatType, typename D, typename FDist, typename FCost>
  [[nodiscard]] FloatType msm(const D& series1, size_t length1, const D& series2, size_t length2,
    FDist dist, FCost msm_cost, const FloatType co, FloatType ub = utils::PINF<double>
  ) {
    const auto check_result = check_order_series<FloatType>(series1, length1, series2, length2);
    switch (check_result.index()) {
      case 0: { return std::get<0>(check_result); }
      case 1: {
        const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        if (std::isinf(ub)) {
          ub = 0;
          // We have less columns than lines: cover all the columns first -- Diag: move
          // Then go down in the last column -- Above: Split/Merge
          for (size_t i{0}; i<nbcols; ++i) { ub += dist(lines, i, cols, i); }
          for (size_t i{nbcols}; i<nblines; ++i) { ub += msm_cost(lines, i, i-1, cols, nbcols-1, co); }
        } else if (std::isnan(ub)) { ub = utils::PINF<FloatType>; }
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::msm<FloatType>(lines, nblines, cols, nbcols, dist, msm_cost, co, ub);
      }
      default: utils::should_not_happen();
    }
  }

  /// Builder for the univariate MSM cost function
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto msm_cost_uni() { return internal::msm_cost_uni<FloatType, D>; }

  /// Helper with a distance builder 'mkdist' and a cost function builder 'mkcost'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType
  msm(const D& s1, const D& s2, auto mkdist, auto mkmid, const FloatType co, FloatType ub) {
    return msm(s1, s1.size(), s2, s2.size(), mkdist(), mkmid(), co, ub);
  }

  /// Helper with the absolute value as the default distance builder,
  /// the default univariate msm cost, and a default ub at +INF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType
  msm(const D& s1, const D& s2, const FloatType co, FloatType ub = utils::PINF<FloatType>) {
    return msm(s1, s1.size(), s2, s2.size(), distance::abs<FloatType, D>(), msm_cost_uni<FloatType, D>(), co, ub);
  }

  /// Builder for the multivariate MSM cost function
  template<typename FloatType, typename D, typename FDist, typename FDistMP>
  [[nodiscard]] inline auto msm_cost_multi(size_t ndim, FDist dist, FDistMP distmpoint) {
    return [ndim, dist, distmpoint](const D& X, size_t xnew_, size_t xi_, const D& Y, size_t yj_, FloatType cost) {
      return internal::msm_cost_multi<FloatType, D, FDist, FDistMP>(
        X, xnew_, xi_, Y, yj_, dist, distmpoint, cost
      );
    };
  }

  /// Helper with a distance builder 'mkdist' and a cost function builder 'mkcost'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType
  msm(const D& s1, const D& s2, size_t ndim, auto mkdist, auto mkmid, const FloatType co, FloatType ub) {
    return msm(s1, s1.size()/ndim, s2, s2.size()/ndim, mkdist(), mkmid(), co, ub);
  }

  /// Helper with the Euclidean distance value as the default distance builder,
  /// the default univariate msm cost, and a default ub at +INF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType
  msm(const D& s1, const D& s2, size_t ndim, const FloatType co, FloatType ub = utils::PINF<FloatType>) {
    return msm(s1, s1.size()/ndim, s2, s2.size()/ndim,
      distance::ed<FloatType, D>(ndim),
      msm_cost_multi<FloatType, D>(ndim, distance::ed<FloatType, D>(ndim), distance::edNmid<FloatType, D>(ndim)),
      co, ub
    );
  }


} // End of namespace libtempo::distance
