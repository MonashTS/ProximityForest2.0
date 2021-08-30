#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

namespace libtempo::distance {

  namespace internal {

    enum FGDIST_SWITCH {
      COLS,
      LINES
    };

    /** Edit Distance with Real Penalty (ERP), with cut-off point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam FDist        Distance computation function, must be a (size_t, size_t)->FloatType
     * @tparam FGDist       Gap distance computation function, must be a (size_t)->FloatType
     * @param nblines       Length of the line series. Must be 0 < nbcols <= nblines
     * @param nbcols        Length of the column series. Must be 0 < nbcols <= nblines
     * @param dist          Distance function, has to capture the series as it only gets the (li,co) coordinate
     * @param gdist_li     "Gap Value" function distance for lines   (i.e. dist(lines, gv))
     * @param gdist_co     "Gap Value" function distance for columns (i.e. dist(gv, cols))
     * @param w             Half-window parameter (looking at w cells on each side of the diagonal)
     *                      Must be 0<=w<=nblines and nblines - nbcols <= w
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @return ERP value or +INF if early abandoned, or , given w, no alignment is possible
     */
    template<typename FloatType, typename FDist, typename FGDist>
    [[nodiscard]] inline FloatType erp(
      size_t nblines, size_t nbcols,
      FDist dist, FGDist gdist_li, FGDist gdist_co,
      size_t w,
      const FloatType cutoff
      ) {

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);
      assert(nbcols<=nblines);
      assert(w<=nblines);
      assert(nblines-nbcols<=w);
      // Adapt constants to the floating point type
      constexpr auto PINF = utils::PINF<FloatType>;
      using utils::min;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const FloatType ub = utils::initBlock{
        const auto la = min(
          gdist_co(nbcols - 1),           // Previous
          dist(nblines - 1, nbcols - 1),  // Diagonal
          gdist_li(nblines - 1)           // Above
        );
        return nextafter(cutoff, PINF) - la;
      };

      return 0.0;
    }

  } // End of namespace internal

} // End of namespace libtempo::distance
