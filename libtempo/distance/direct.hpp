#pragma once

#include <libtempo/utils/utils.hpp>
#include "cost_function.hpp"
#include <libtempo/concepts.hpp>

#include <cmath>

namespace libtempo::distance {

  namespace internal {

    /** Direct alignment (Euclidean distance like) with early abandoning. No pruning.
     * Only defined for same length series (return +INF if different length).
     * @tparam FloatType  The floating number type used to represent the series.
     * @tparam D          Type of underlying collection - given to dist
     * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
     * @param length1     Length of the first series.
     * @param length2     Length of the second series.
     * @param dist        Distance function of type FDist
     * @param cutoff.     Attempt to prune computation of alignments with cost > cutoff.
     *                    May lead to early abandoning.
     * @return Norm according to 'dist' between the two series, or +INF if early abandoned.
     */
    template<Float F>
    [[nodiscard]] inline F
    directa(
      const size_t length1,
      const size_t length2,
      CFun<F> auto dist,
      const F cutoff
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      constexpr auto PINF = utils::PINF<F>;
      // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
      if (length1!=length2) { return PINF; }
      if (length1==0) { return 0; }
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      // Adjust the lower bound, taking the last alignment into account
      const F lastA = dist(length1-1, length1-1);
      const F ub = std::nextafter(cutoff, PINF)-lastA;
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Compute the Euclidean-like distance up to, excluding, the last alignment
      double cost = 0;
      for (size_t i{0}; i<length1-1; ++i) { // Stop before the last: counted in the bound!
        cost += dist(i, i);
        if (cost>ub) { return PINF; }
      }
      // Add the last alignment and check the result
      cost += lastA;
      if (cost>cutoff) { return PINF; } else { return cost; }
    }

    /** Direct alignment (Euclidean distance like). No early abandoning. No pruning.
     * Only defined for same length series (return +INF if different length).
     * @tparam FloatType  The floating number type used to represent the series.
     * @tparam D          Type of underlying collection - given to dist
     * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
     * @param length1     Length of the first series.
     * @param length2     Length of the second series.
     * @param dist        Distance function of type FDist
     * @return Norm according to 'dist' between the two series
     */
    template<Float F>
    [[nodiscard]] inline F directa(const size_t length1, const size_t length2, CFun<F> auto dist) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Max error on different length.
      if (length1!=length2) { return utils::PINF<F>; }
      // Compute the Euclidean-like distance
      F cost = 0.0;
      for (size_t i{0}; i<length1; ++i) { cost += dist(i, i); }
      return cost;
    }
  } // End of namespace internal


  /** Direct alignment distance. Can early abandon but not prune.
   * Return +INF if the series have different length.
   * @tparam FloatType  The floating number type used to represent the series.
   * @tparam D          Type of underlying collection - given to dist
   * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
   * @param length1     Length of the first series.
   * @param length2     Length of the second series.
   * @param dist        Distance function, has to capture the series as it only gets the (li,co) coordinates
   * @param ub          Upper bound. Attempt to prune computation of alignments with cost > cutoff.
   *                    May lead to early abandoning.
   *                    ub = PINF: no lower bounding
   *                    ub = QNAN: no lower bounding
   *                    ub = other value: use for pruning and early abandoning
   * @return Norm according to 'dist' between the two series
  */
  template<Float F>
  [[nodiscard]] F
  directa(const size_t length1, const size_t length2, CFun<F> auto dist, F ub) {
    if (std::isinf(ub) || std::isnan(ub)) {
      return internal::directa<F>(length1, length2, dist);
    } else { return internal::directa<F>(length1, length2, dist, ub); }
  }

  /// Helper for TSLike
  template<Float F, TSLike T>
  [[nodiscard]] inline F
  directa(const T& lines, const T& cols, CFunBuilder<T> auto mkdist, F ub = utils::PINF<F>) {
    const auto ls = lines.length();
    const auto cs = cols.length();
    const CFun<F> auto dist = mkdist(lines, cols);
    return directa<F>(ls, cs, dist, ub);
  }

  namespace univariate {

    /// Default, using univariate ad2
    template<Float F, TSLike T>
    [[nodiscard]] inline F directa(const T& lines, const T& cols, F ub = utils::PINF<F>) {
      return directa(lines, cols, ad2<F, T>, ub);
    }

    /// Specific overload for univariate vector
    template<Float F>
    [[nodiscard]] inline F
    directa(const std::vector<F>& lines, const std::vector<F>& cols, CFunBuilder<std::vector<F>> auto mkdist,
      F ub = utils::PINF<F>) {
      const auto ls = lines.size();
      const auto cs = cols.size();
      const CFun<F> auto dist = mkdist(lines, cols);
      std::vector<F> v;
      return libtempo::distance::directa<F>(ls, cs, dist, ub);
    }

    /// Specific overload for univariate vector
    template<Float F>
    [[nodiscard]] inline F directa(const std::vector<F>& lines, const std::vector<F>& cols, F ub = utils::PINF<F>) {
      return directa<F>(lines, cols, ad2<F, std::vector<F>>, ub);
    }

  }


} // End of namespace libtempo::distance
