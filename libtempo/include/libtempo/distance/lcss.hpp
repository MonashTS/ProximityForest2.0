#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

namespace libtempo::distance {

  namespace lcss_details {

    /// Check if two FloatType numbers are within epsilon (true = similar) - used by LCSS
    template<typename FloatType, typename D>
    [[nodiscard]] bool sim(const D& lines, size_t li, const D& cols, size_t co, FloatType epsilon) {
      return std::fabs(lines[li]-cols[co])<epsilon;
    }

    /// Check if two FloatType numbers are within epsilon (true = similar) - used by LCSS
    template<typename FloatType, typename D>
    [[nodiscard]] bool simN(const D& lines, size_t li, const D& cols, size_t co, FloatType epsilon, size_t ndim) {
      return sqedN<FloatType, D>(lines, li, cols, co, ndim)<epsilon;
    }
  }

  /// Get the univariate simple similarity measure FSim for LCSS
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sim() { return lcss_details::sim<FloatType, D>; }

  /// Get the multivariate simple similarity measure FSim for LCSS
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sim(size_t ndim) {
    return [ndim](const D& X, size_t xnew, size_t xi, const D& Y, size_t yi) {
      return lcss_details::simN<FloatType, D>(X, xnew, xi, Y, yi, ndim);
    };
  }

  /** Longest Common SubSequence (LCSS) with early abandoning.
   *  Double buffered implementation using O(n) space.
   *  Note: no pruning, only early abandoning (not the same structure as other "dtw-like" distances)
   * @tparam FloatType    The floating number type used to represent the series.
   * @tparam D            Type of underlying collection - given to dist
   * @tparam FSim         Similarity computation function, must be a (const D&, size_t, constD&, size_t, FloatType)->bool
   * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
   * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
   * @param sim           Similarity function of type FSim
   * @param epsilon       Threshold comparison the sim function
   * @param w             Half-window parameter (looking at w cells on each side of the diagonal)
   *                      Must be 0<=w<=nblines and nblines - nbcols <= w
   * @param ub            Upper bound. Attempt to early abandon computation of alignments with cost > ub.
   *                      ub = PINF or QNAN or >1: no early abandoning
   *                      ub = other value: use for early abandoning - capped to 1
   * @return LCSS dissimilarity measure [0,1] where 0 stands for identical series and 1 completely distinct.
   *         +INF id early abandoned or no alignment is possible given the window w and the length of the series.
   */
  template<typename FloatType, typename D, typename FSim>
  [[nodiscard]] FloatType lcss(
    const D& series1, size_t length1,
    const D& series2, size_t length2,
    FSim sim, FloatType epsilon, size_t w,
    FloatType ub = utils::PINF<double>
  ) {
    using namespace utils;
    constexpr auto PINF = utils::PINF<FloatType>;

    const auto check_result = check_order_series<FloatType>(series1, length1, series2, length2);
    switch (check_result.index()) {
      case 0: { return std::get<0>(check_result); }
      case 1: {
        const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
        // Cap the windows and check that, given the constraint, an alignment is possible
        if (w>nblines) { w = nblines; }
        if (nblines-nbcols>w) { return PINF; }
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Double buffer allocation, init to 0.
        // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
        // Initialisation OK as is: border line and "first diag" init to 0
        std::vector<size_t> buffers_v((1+nbcols)*2, 0);
        size_t* buffers = buffers_v.data();
        size_t c{0+1}, p{nbcols+2};
        if (ub>1 || std::isnan(ub)) {
          // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
          // NO EA
          for (size_t i{0}; i<nblines; ++i) {
            // --- --- --- Swap and variables init
            std::swap(c, p);
            const size_t jStart = cap_start_index_to_window(i, w);
            const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
            // --- --- --- Init the border (very first column)
            buffers[c+jStart-1] = 0;
            // --- --- --- Iterate through the columns
            for (size_t j{jStart}; j<jStop; ++j) {
              if (sim(lines, i, cols, j, epsilon)) { buffers[c+j] = buffers[p+j-1]+1; } // Diag + 1
              else { // Note: Diagonal lookup required, e.g. when w=0
                buffers[c+j] = max(buffers[c+j-1], buffers[p+j-1], buffers[p+j]);
              }
            } // End for loop j
          } // End for loop i
        } else {
          // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
          // WITH EA
          ub = std::max<FloatType>(0, ub);
          const size_t to_reach = std::ceil((1-ub)*nbcols);
          size_t current_max = 0;
          for (size_t i{0}; i<nblines; ++i) {
            // --- --- --- Stop if not enough remaining lines to reach the target (by taking the diagonal)
            const size_t lines_left = nblines-i;
            if (current_max+lines_left<to_reach) { return PINF; }
            // --- --- --- Swap and variables init
            std::swap(c, p);
            const size_t jStart = cap_start_index_to_window(i, w);
            const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
            // --- --- --- Init the border (very first column)
            buffers[c+jStart-1] = 0;
            // --- --- --- Iterate through the columns
            for (size_t j{jStart}; j<jStop; ++j) {
              if (sim(lines, i, cols, j, epsilon)) { // Diag + 1
                const size_t cost = buffers[p+j-1]+1;
                current_max = std::max(current_max, cost);
                buffers[c+j] = cost;
              } // Note: Diagonal lookup required, e.g. when w=0
              else { buffers[c+j] = max(buffers[c+j-1], buffers[p+j-1], buffers[p+j]); }
            } // End for loop j
          } // End for loop i
        } // End EA
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Finalisation: put the result on a [0 - 1] range
        return 1.0-(double(buffers[c+nbcols-1])/(double) nbcols);
      } // End case 1
      default: utils::should_not_happen();
    }
  }

  /// Helper requiring a similarity function builder 'mksim'
  template<typename FloatType>
  [[nodiscard]] inline FloatType lcss(
    const auto& s1, const auto& s2, auto mksim, FloatType epsilon, size_t w, FloatType ub
  ) { return lcss(s1, s1.size(), s2, s2.size(), mksim(), w, epsilon, ub); }

  /// Helper with a default similarity function builder, and a default ub at +INF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType lcss(
    const D& s1, const D& s2, FloatType epsilon, size_t w, FloatType ub = utils::PINF<FloatType>
  ) { return lcss(s1, s1.size(), s2, s2.size(), sim<FloatType, D>(), epsilon, w, ub); }

  /// Multivariate helper requiring a similarity function builder 'mksim'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType lcss(
    const D& s1, const D& s2, size_t ndim, auto mksim, FloatType epsilon, size_t w, FloatType ub
  ) { return lcss(s1, s1.size()/ndim, s2, s2.size()/ndim, mksim(ndim), epsilon, w, ub); }

  /// Multidimensional helper with a default similarity function builder, and a default ub at +INF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType lcss(
    const D& s1, const D& s2, size_t ndim, FloatType epsilon, size_t w, FloatType ub = utils::PINF<FloatType>
  ) { return lcss(s1, s1.size()/ndim, s2, s2.size()/ndim, sim<FloatType, D>(ndim), epsilon, w, ub); }


} // End of namespace libtempo::distance
