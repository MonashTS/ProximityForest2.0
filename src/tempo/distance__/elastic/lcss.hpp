#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/distance__/helpers.hpp>

namespace tempo::distance {

  /// LCSS specific cost function concept, assessing if two points (represented by their index) are similar or not
  /// Both series must be captured
  template<typename Fun>
  concept CFunSim = requires(Fun fun, size_t i, size_t j){
    { fun(i, j) }->std::same_as<bool>;
  };

  /** Longest Common SubSequence (LCSS) with early abandoning.
   *  Double buffered implementation using O(n) space.
   *  Note: no pruning, only early abandoning (not the same structure as other "dtw-like" distances)
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
  [[nodiscard]] F lcss(const size_t nblines,
                       const size_t nbcols,
                       CFunSim auto sim,
                       const size_t w,
                       F ub
  ) {
    using namespace utils;
    constexpr F PINF = utils::PINF;
    if (nblines==0&&nbcols==0) { return 0; }
    else if ((nblines==0)!=(nbcols==0)) { return PINF; }
    else {
      // Check that the window allows for an alignment
      // If this is accepted, we do not need to check the window when computing a new UB
      const auto m = std::min(nblines, nbcols);
      const auto M = std::max(nblines, nbcols);
      if (M - m>w) { return PINF; }
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, init to 0.
      // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
      // Initialisation OK as is: border line and "first diag" init to 0
      std::vector<size_t> buffers_v((1 + nbcols)*2, 0);
      size_t *buffers = buffers_v.data();
      size_t c{0 + 1}, p{nbcols + 2};
      // Do we need to EA?
      if (ub>1||std::isnan(ub)||std::isinf(ub)) { // Explicitely catch inf
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // NO EA
        for (size_t i{0}; i<nblines; ++i) {
          // --- --- --- Swap and variables init
          std::swap(c, p);
          const size_t jStart = cap_start_index_to_window(i, w);
          const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
          // --- --- --- Init the border (very first column)
          buffers[c + jStart - 1] = 0;
          // --- --- --- Iterate through the columns
          for (size_t j{jStart}; j<jStop; ++j) {
            if (sim(i, j)) { buffers[c + j] = buffers[p + j - 1] + 1; } // Diag + 1
            else { // Note: Diagonal lookup required, e.g. when w=0
              buffers[c + j] = max(buffers[c + j - 1], buffers[p + j - 1], buffers[p + j]);
            }
          } // End for loop j
        } // End for loop i
      } else {
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // WITH EA
        ub = std::max<F>(0.0, ub);
        const size_t to_reach = std::ceil((1 - ub)*m); // min value here
        size_t current_max = 0;
        for (size_t i{0}; i<nblines; ++i) {
          // --- --- --- Stop if not enough remaining lines to reach the target (by taking the diagonal)
          const size_t lines_left = nblines - i;
          if (current_max + lines_left<to_reach) { return PINF; }
          // --- --- --- Swap and variables init
          std::swap(c, p);
          const size_t jStart = cap_start_index_to_window(i, w);
          const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
          // --- --- --- Init the border (very first column)
          buffers[c + jStart - 1] = 0;
          // --- --- --- Iterate through the columns
          for (size_t j{jStart}; j<jStop; ++j) {
            if (sim(i, j)) { // Diag + 1
              const size_t cost = buffers[p + j - 1] + 1;
              current_max = std::max(current_max, cost);
              buffers[c + j] = cost;
            } // Note: Diagonal lookup required, e.g. when w=0
            else { buffers[c + j] = max(buffers[c + j - 1], buffers[p + j - 1], buffers[p + j]); }
          } // End for loop j
        } // End for loop i
      } // End EA
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation: put the result on a [0 - 1] range - normalize by the minimum value (max +1 we can do)
      return 1.0 - (double(buffers[c + nbcols - 1])/(double)m);
    }
  }

  /// Helper with a similarity function based on epsilon (dist(a,b)<e)
  [[nodiscard]] inline F lcss(size_t nblines, size_t nbcols, CFun auto dist, const size_t w, const F e, F ub) {
    CFunSim auto sim = [dist, e](size_t i, size_t j) { return dist(i, j)<e; };
    return lcss(nblines, nbcols, sim, w, ub);
  }

  /// Helper for TSLike, with a similarity function based on epsilon (dist(a,b)<e)
  template<TSLike T>
  [[nodiscard]] inline F
  lcss(const T& lines, const T& cols, CFunBuilder<T> auto mkdist, const size_t w, const F e, F ub) {
    const auto ls = lines.length();
    const auto cs = cols.length();
    const CFun auto dist = mkdist(lines, cols);
    CFunSim auto sim = [dist, e](size_t i, size_t j) { return dist(i, j)<e; };
    return lcss(ls, cs, sim, w, ub);
  }

  namespace multivariate {

    template<Subscriptable D>
    [[nodiscard]] inline auto simN(const D& lines, const D& cols, size_t ndim, F e) {
      return [&, ndim, e](size_t i, size_t j) { return ad2N(lines, cols, ndim)(i, j)<e; };
    }

  }



  /*
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

  */

} // End of namespace tempo::distance





/*


  namespace lcss_details {

    /// Check if two FloatType numbers are within epsilon (true = similar) - used by LCSS
    template<typename FloatType, typename D>
    [[nodiscard]] bool sim(const D& lines, size_t li, const D& cols, size_t co, FloatType epsilon) {
      return std::fabs(lines[li]-cols[co])<epsilon;
    }

    /// Check if two FloatType numbers are within epsilon (true = similar) - used by LCSS
    template<typename FloatType, typename D>
    [[nodiscard]] bool simN(const D& lines, size_t li, const D& cols, size_t co, size_t ndim, FloatType epsilon) {
      return internal::sqedN<FloatType, D>(lines, li, cols, co, ndim)<epsilon;
    }
  }

  /// Get the univariate simple similarity measure FSim for LCSS
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sim() { return lcss_details::sim<FloatType, D>; }

  /// Get the multivariate simple similarity measure FSim for LCSS
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sim(size_t ndim) {
    return [ndim](const D& X, size_t xi, const D& Y, size_t yi, FloatType epsilon) {
      return lcss_details::simN<FloatType, D>(X, xi, Y, yi, ndim, epsilon);
    };
  }


 */