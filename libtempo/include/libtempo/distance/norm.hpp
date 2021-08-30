#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/distance/distance.hpp>

#include <cmath>

namespace libtempo::distance {

  namespace internal {

    /** Usual norm, Euclidean distance like distance with early abandoning. No pruning.
     * Only defined for same length series (return +INF if different length).
     * @tparam FloatType  The floating number type used to represent the series.
     * @tparam D          Type of underlying collection - given to dist
     * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
     * @param length1     Length of the first series.
     * @param length2     Length of the second series.
     * @param dist        Distance function, has to capture the series as it only gets the (li,co) coordinate
     * @param cutoff.     Attempt to prune computation of alignments with cost > cutoff.
     *                    May lead to early abandoning.
     * @return Norm according to 'dist' between the two series, or +INF if early abandoned.
     */
    template<typename FloatType, typename D, typename FDist>
    [[nodiscard]] inline FloatType
    norm(const D& s1, size_t length1, const D& s2, size_t length2, FDist dist, const FloatType cutoff) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      constexpr auto PINF = utils::PINF<FloatType>;
      // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
      if (length1!=length2) { return PINF; }
      if (length1==0) { return 0; }
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      // Adjust the lower bound, taking the last alignment into account
      const FloatType lastA = dist(s1, length1-1, s2, length1-1);
      const FloatType ub = std::nextafter(cutoff, PINF)-lastA;
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Compute the Euclidean-like distance up to, excluding, the last alignment
      double cost = 0;
      for (size_t i{0}; i<length1-1; ++i) { // Stop before the last: counted in the bound!
        cost += dist(s1, i, s2, i);
        if (cost>ub) { return PINF; }
      }
      // Add the last alignment and check the result
      cost += lastA;
      if (cost>cutoff) { return PINF; } else { return cost; }
    }

    /** Usual norm, Euclidean distance like distance. No early abandoning. No pruning.
     * Only defined for same length series (return +INF if different length).
     * @tparam FloatType  The floating number type used to represent the series.
     * @tparam D          Type of underlying collection - given to dist
     * @tparam FDist      Distance computation function, must be a (size_t, size_t)->FloatType
     * @param length1     Length of the first series.
     * @param length2     Length of the second series.
     * @param dist        Distance function, has to capture the series as it only gets the (li,co) coordinate
     * @return Norm according to 'dist' between the two series
     */
    template<typename FloatType, typename D, typename FDist>
    [[nodiscard]] inline FloatType norm(const D& s1, size_t length1, const D& s2, size_t length2, FDist dist) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Max error on different length.
      if (length1!=length2) { return utils::PINF<FloatType>; }
      // Compute the Euclidean-like distance
      FloatType cost = 0.0;
      for (size_t i{0}; i<length1; ++i) { cost += dist(s1, i, s2, i); }
      return cost;
    }
  } // End of namespace internal




  /** Norm according to a point-to-point distance. Can early abandon but not prune.
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
  template<typename FloatType, typename D, typename FDist>
  [[nodiscard]] FloatType
  norm(const D& series1, size_t length1, const D& series2, size_t length2, FDist dist, FloatType ub) {
    if (std::isinf(ub) || std::isnan(ub)) {
      return internal::norm<FloatType, D, FDist>(series1, length1, series2, length2, dist);
    }
    else { return internal::norm<FloatType, D, FDist>(series1, length1, series2, length2, dist, ub); }
  }

  /// Helper with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType norm(const D& s1, const D& s2, auto mkdist, FloatType ub) {
    return norm(s1, s1.size(), s2, s2.size(), mkdist(), ub);
  }

  /// Helper with the sqed as the default distance builder, and a default ub at +INF
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType norm(const D& s1, const D& s2, FloatType ub = utils::PINF<FloatType>) {
    return norm(s1, s1.size(), s2, s2.size(), distance::sqed<FloatType, D>(), ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType norm(const D& s1, const D& s2, size_t ndim, auto mkdist, FloatType ub) {
    return norm(s1, s1.size()/ndim, s2, s2.size()/ndim, mkdist(ndim), ub);
  }

  /// Multidimensional helper, with a distance builder 'mkdist'
  template<typename FloatType, typename D>
  [[nodiscard]] inline FloatType norm(const D& s1, const D& s2, size_t ndim, FloatType ub = utils::PINF<FloatType>) {
    return norm(s1, s1.size()/ndim, s2, s2.size()/ndim, distance::sqed<FloatType, D>(ndim), ub);
  }


} // End of namespace libtempo::distance
