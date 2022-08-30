#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/distance__/helpers.hpp>

namespace tempo::distance {

  /// ERP specific cost function concept, computing the distance to a point (represented by its index)
  /// to the "gap value".
  /// Both the series and the gap values must be captured
  template<typename Fun>
  concept CFunGV = requires(Fun fun, size_t i){
    { fun(i) }->std::convertible_to<F>;
  };

  /// CVFunGVBuilder: Function creating a CFunGV based on a series and a gap value
  template<typename T, typename D>
  concept CFunGVBuilder = requires(T builder, const D& s, const F gv){
    builder(s, gv);
  };

  namespace internal {

    /* The gap value function must implement something like this for Previous (gv_cols) and Above (gv_lines)
    dist(gValue, cols[j]),    // Previous
    dist(lines[i], cols[j]),  // Diagonal
    dist(lines[i], gValue)    // Above
     */

    /** Edit Distance with Real Penalty (ERP), with cut-off point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
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
    [[nodiscard]] inline F erp(const size_t nblines,
                               const size_t nbcols,
                               CFunGV auto dist_gv_lines,
                               CFunGV auto dist_gv_cols,
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
      using utils::PINF;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = utils::initBlock {
        const auto la = min(
          dist_gv_cols(nbcols - 1),      // Previous (col)
          dist(nblines - 1, nbcols - 1),   // Diagonal
          dist_gv_lines(nblines - 1)     // Above    (line)
        );
        return nextafter(cutoff, PINF) - la;
      };

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
      // Initialisation of the top border
      {   // Matrix Border - Top diagonal
        buffer[c - 1] = 0;
        // Matrix Border - First line
        const size_t jStop = cap_stop_index_to_window_or_end(0, w, nbcols);
        for (j = 0; buffer[c + j - 1]<=ub&&j<jStop; ++j) {
          buffer[c + j] = buffer[c + j - 1] + dist_gv_cols(j); // Previous
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
            cost = buffer[c - 1] + dist_gv_lines(i);
            if (cost>ub) { break; }
            else {
              std::swap(c, p);
              buffer[c - 1] = cost;
            }
          }
          // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
          // No stage 1 here.
          // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
          for (; j<prev_pp; ++j) {
            cost = min(
              cost + dist_gv_cols(j),           // Previous
              buffer[p + j - 1] + dist(i, j),       // Diagonal
              buffer[p + j] + dist_gv_lines(i)    // Above
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
          if (j<jStop) { // Possible path in previous cells: left and diag.
            cost = std::min(
              cost + dist_gv_cols(j),       // Previous
              buffer[p + j - 1] + dist(i, j)    // Diagonal
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            ++j;
          }
          // --- --- --- Stage 4: After the previous pruning point: only prev.
          // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
          for (; j==curr_pp&&j<jStop; ++j) {
            cost = cost + dist_gv_cols(j);  // Previous
            buffer[c + j] = cost;
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
            buffer[c + jStart - 1] = cost;
          }
          // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
          for (; j==next_start&&j<prev_pp; ++j) {
            cost = std::min(
              buffer[p + j - 1] + dist(i, j),       // Diagonal
              buffer[p + j] + dist_gv_lines(i)    // Above
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          }
          // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
          for (; j<prev_pp; ++j) {
            cost = min(
              cost + dist_gv_cols(j),         // Previous
              buffer[p + j - 1] + dist(i, j),     // Diagonal
              buffer[p + j] + dist_gv_lines(i)  // Above
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
          if (j<jStop) { // If so, two cases.
            if (j==next_start) { // Case 1: Advancing next start: only diag.
              cost = buffer[p + j - 1] + dist(i, j);     // Diagonal
              buffer[c + j] = cost;
              if (cost<=ub) { curr_pp = j + 1; }
              else {
                // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; } else { return PINF; }
              }
            } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
              cost = std::min(
                cost + dist_gv_cols(j),     // Previous
                buffer[p + j - 1] + dist(i, j)  // Diagonal
              );
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
          for (; j==curr_pp&&j<jStop; ++j) {
            cost = cost + dist_gv_cols(j);
            buffer[c + j] = cost;
            if (cost<=ub) { ++curr_pp; }
          }
          // --- --- ---
          prev_pp = curr_pp;
        }
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols&&cost<=cutoff) { return cost; } else { return PINF; }
    }

  } // End of namespace internal

  /** Edit Distance with Real Penalty (ERP), with cut-off point for early abandoning and pruning.
   *  Double buffered implementation using O(n) space.
   *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
   *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
   *  Actual implementation assuming that some pre-conditions are fulfilled.
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
  [[nodiscard]] F erp(const size_t nblines,
                      const size_t nbcols,
                      CFunGV auto dist_gv_lines,
                      CFunGV auto dist_gv_cols,
                      CFun auto dist,
                      const size_t w,
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
        // Fewer line than columns: complete the last line (advancing in the columns)
        if (nblines<nbcols) { for (size_t i{nblines}; i<nbcols; ++i) { ub = ub + dist_gv_cols(i); }}
          // Fewer columns than lines: complete the last column (advancing in the lines)
        else if (nbcols<nblines) { for (size_t i{nbcols}; i<nblines; ++i) { ub = ub + dist_gv_lines(i); }}
      } else if (std::isnan(ub)) { ub = PINF; }
      // ub computed
      return internal::erp(nblines, nbcols, dist_gv_lines, dist_gv_cols, dist, w, ub, buffer_v);
    }
  }

  /// Helper without having to provide a buffer
  [[nodiscard]] inline F erp(const size_t nblines,
                             const size_t nbcols,
                             CFunGV auto dist_gv_lines,
                             CFunGV auto dist_gv_cols,
                             CFun auto dist,
                             const size_t w,
                             F ub
  ) {
    std::vector<F> v;
    return erp(nblines, nbcols, dist_gv_lines, dist_gv_cols, dist, w, ub, v);
  }

  /// Helper for TSLike, without having to provide a buffer
  template<TSLike T>
  [[nodiscard]] inline F erp(const T& lines,
                             const T& cols,
                             CFunGVBuilder<T> auto mkdist_gv,
                             CFunBuilder<T> auto mkdist,
                             const size_t w,
                             const F gv,
                             F ub
  ) {
    const auto ls = lines.length();
    const auto cs = cols.length();
    const CFunGV auto lines_gv = mkdist_gv(lines, gv);
    const CFunGV auto cols_gv = mkdist_gv(cols, gv);
    const CFun auto dist = mkdist(lines, cols);
    return erp(ls, cs, lines_gv, cols_gv, dist, w, ub);
  }

  namespace univariate {

    /// CFunGVBuilder Univariate Absolute difference exponent 1
    template<Subscriptable D>
    auto ad1gv(const D& series, const F gv) {
      return [&, gv](size_t i) {
        const F d = series[i] - gv;
        return std::abs(d);
      };
    }

    /// CFunGVBuilder Univariate Absolute difference exponent 2
    template<Subscriptable D>
    auto inline ad2gv(const D& series, const F gv) {
      return [&, gv](size_t i) {
        const F d = series[i] - gv;
        return d*d;
      };
    }

    /// CFunGVBuilder Univariate Absolute difference exponent e
    template<Subscriptable D>
    auto inline adegv(const F e) {
      return [e](const D& series, F gv) {
        return [&, gv, e](size_t i) {
          const F d = std::abs(series[i] - gv);
          return std::pow(d, e);
        };
      };
    }
  }

  namespace multivariate {

    /// WARP distance based on the squared absolute difference and a vector for gv
    /// Dimension is the same as the size of gv (all dimensions used)
    template<Subscriptable D>
    inline auto ad2gv(const D& series, const std::vector<F>& gv) {
      const auto ndim = gv.size();
      return [&, ndim](size_t i) {
        const size_t offset = i*ndim;
        F acc{0};
        for (size_t k{0}; k<ndim; ++k) {
          const F d = series[offset + k] - gv[k];
          acc += d*d;
        }
        return acc;
      };
    };

  }
} // End of namespace tempo::distance
