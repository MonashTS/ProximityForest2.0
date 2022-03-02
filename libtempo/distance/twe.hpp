#pragma once

#include <libtempo/utils/utils.hpp>
#include "cost_function.hpp"

namespace libtempo::distance {

  namespace internal {

    /** Edit Distance with Real Penalty (MSM), with cut-off point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @tparam FDist        Distance computation function, must be a (const D&, size_t, constD&, size_t)->FloatType
     * @tparam FCost        MSM cost computation function - see twe_cost_uni and msms_cost_multi
     * @param lines         Data for the lines
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param cols          Data for the lines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist          Distance function of type FDist
     * @param nu            Stiffness parameter
     * @param lambda        Penalty parameter
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return MSM value or +INF if early abandoned, or , given w, no alignment is possible
     */
    template<typename FloatType, typename D, typename FDist>
    [[nodiscard]] inline FloatType twe(
      const D& lines, size_t nblines,
      const D& cols, size_t nbcols,
      FDist dist,
      const FloatType nu, const FloatType lambda,
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

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Constants: we only consider timestamp spaced by 1, so:
      // In the "delete" case, we always have a time difference of 1, so we always have 1*nu+lambda
      const auto nu_lambda = nu+lambda;
      // In the "match" case, we always have nu*(|i-j|+|(i-1)-(j-1)|) == 2*nu*|i-j|
      const auto nu2 = FloatType(2)*nu;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const FloatType ub = initBlock {
        // The last alignment can only computed if we have nbcols >= 2
        if (nbcols>=2) {
          const auto la = min(
            // "Delete_B": over the columns / Prev
            dist(cols, nbcols-1, cols, nbcols-1)+nu_lambda,
            // Match: Diag. Ok: nblines >= nbcols
            dist(lines, nblines-1, cols, nbcols-1)+dist(lines, nblines-2, cols, nbcols-2)+nu2*(nblines-nbcols),
            // "Delete_A": over the lines / Top
            dist(lines, nblines-2, lines, nblines-1)+nu_lambda
          );
          return FloatType(nextafter(cutoff, PINF)-la);
        } else { return FloatType(cutoff); }
      };

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      auto buffers = std::unique_ptr<FloatType[]>(new FloatType[nbcols*2]);
      size_t c{0}, p{nbcols};

      // Buffer holding precomputed distance between columns
      auto distcol = std::unique_ptr<FloatType[]>(new FloatType[nbcols]);

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      FloatType cost;

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the first line. Deal with the line top border condition.
      {
        // Case [0,0]: special "Match case"
        cost = dist(lines, 0, cols, 0);
        buffers[c+0] = cost;
        // Distance for the first column is relative to 0 "by conventions" (from the paper, section 4.2)
        // Something like
        // distcol[0]=dist(0, cols[0]);
        // However, this would only be use for j=0, i.e. with the left border which is +INF,
        // hence the result would always be +INF: we can simply use 0 instead.
        // Note that the border is managed as part of the code, hence we actually never access distcol[0]!
        // distcol[0] = 0;
        // Rest of the line: [i==0, j>=1]: "Delete_B case" (prev)
        // We also initialize 'distcol' here.
        for (j = 1; j<nbcols; ++j) {
          const double d = dist(cols, j-1, cols, j);
          distcol[j] = d;
          cost = cost+d+nu_lambda;
          buffers[c+j] = cost;
          if (cost<=ub) { prev_pp = j+1; } else { break; }
        }
        // Complete the initialisation of distcol
        for (; j<nbcols; ++j) {
          const double d = dist(cols, j-1, cols, j);
          distcol[j] = d;
        }
        // Next line.
        ++i;
      }


      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop, starts at the second line
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        const double distli = dist(lines, i-1, lines, i);
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffers[p+j]+distli+nu_lambda; // "Delete_A" / Top
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start && j<prev_pp; ++j) {
          cost = std::min(
            buffers[p+j-1]+dist(lines, i, cols, j)+dist(lines, i-1, cols, j-1)+nu2*absdiff(i, j), // "Match" / Diag
            buffers[p+j]+distli+nu_lambda // "Delete_A" / Top
          );
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(
            cost+distcol[j]+nu_lambda,      // "Delete_B": over the columns / Prev
            buffers[p+j-1]+dist(lines, i, cols, j)+dist(lines, i-1, cols, j-1)+nu2*absdiff(i, j), // Match: Diag
            buffers[p+j]+distli+nu_lambda   // "Delete_A": over the lines / Top
          );
          buffers[c+j] = cost;
          if (cost<=ub) { curr_pp = j+1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffers[p+j-1]+dist(lines, i, cols, j)+dist(lines, i-1, cols, j-1)+nu2*absdiff(i, j); // Match: Diag
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines-1 && j==nbcols-1 && cost<=cutoff) { return cost; } else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(
              cost+distcol[j]+nu_lambda,     // "Delete_B": over the columns / Prev
              buffers[p+j-1]+dist(lines, i, cols, j)+dist(lines, i-1, cols, j-1)+nu2*absdiff(i, j) // Match: Diag
            );
            buffers[c+j] = cost;
            if (cost<=ub) { curr_pp = j+1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB. Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; } else { next_start = nbcols-1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp && j<nbcols; ++j) {
          cost = cost+distcol[j]+nu_lambda; // "Delete_B": over the columns / Prev
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
   * @tparam VecLike    Vector like datatype - type of "weights", accessed with [index]
   * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
   * @param length1     Length of the first series.
   * @param length2     Length of the second series.
   * @param dist        Distance function of type FDist
   * @param nu          Stiffness parameter
   * @param lambda      Penalty parameter
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    If not provided, defaults to PINF,
   *                    which triggers the computation of an upper bound based on the diagonal.
   *                    Use QNAN to run without any upper bounding.
   * @return DTW between the two series
   */
  template<typename FloatType, typename D, typename FDist>
  [[nodiscard]] FloatType inline twe(
    const D& series1, size_t length1,
    const D& series2, size_t length2,
    FDist dist,
    const FloatType nu, const FloatType lambda,
    FloatType ub = utils::PINF<double>
  ) {
    const auto check_result = check_order_series<FloatType>(series1, length1, series2, length2);
    switch (check_result.index()) {
      case 0: { return std::get<0>(check_result); }
      case 1: {
        const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        if (ub==utils::PINF<FloatType>) {
          const auto nu_lambda = nu+lambda;
          // Compute a cutoff point using the diagonal. Init with the first cell at (0,0)
          ub = dist(lines, 0, cols, 0);
          // We have less columns than lines: cover all the columns first. Starts at (1,1)
          // Match: diagonal with absdiff(i,i) == 0
          // TODO: optimize as we reuse (i, i) in the next step as (i-1, i-1)
          for (size_t i{1}; i<nbcols; ++i) {
            ub = ub+dist(lines, i, cols, i)+dist(lines, i-1, cols, i-1);
          }
          // Then go down in the last column
          for (size_t i{nbcols}; i<nblines; ++i) { ub = ub+dist(lines, i-1, lines, i)+nu_lambda; }
          // A bit of wiggle room due to floats rounding
          ub = nextafter(ub, utils::PINF<FloatType>);
        } else if (std::isnan(ub)) { ub = utils::PINF<FloatType>; }
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::twe<FloatType, D>(lines, nblines, cols, nbcols, dist, nu, lambda, ub);
      }
      default: utils::should_not_happen();
    }
  }

  /// Helper with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType twe(
    const D& s1, const D& s2, auto mkdist,
    const FloatType nu, const FloatType lambda,
    FloatType ub
  ) {
    return twe(s1, s1.size(), s2, s2.size(), mkdist(), nu, lambda, ub);
  }

  /// Helper with the sqed as the default distance builder, and a default ub at +INF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType twe(
    const D& s1, const D& s2,
    const FloatType nu, const FloatType lambda,
    FloatType ub = utils::PINF<FloatType>
  ) {
    return twe(s1, s1.size(), s2, s2.size(), distance::sqed<FloatType, D>(), nu, lambda, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType>
  [[nodiscard]] inline FloatType twe(
    const auto& s1, const auto& s2, size_t ndim, auto mkdist,
    const FloatType nu, const FloatType lambda,
    FloatType ub
  ) {
    return twe(s1, s1.size()/ndim, s2, s2.size()/ndim, mkdist(ndim), nu, lambda, ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType twe(
    const D& s1, const D& s2, size_t ndim,
    const FloatType nu, const FloatType lambda,
    FloatType ub = utils::PINF<FloatType>
  ) {
    return twe(s1, s1.size()/ndim, s2, s2.size()/ndim, distance::sqed<FloatType, D>(ndim), nu, lambda, ub);
  }


} // End of namespace libtempo::distance
