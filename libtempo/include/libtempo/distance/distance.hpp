#pragma once

#include <libtempo/utils/utils.hpp>
#include <libtempo/utils/point.hpp>

#include <functional>
#include <variant>
#include <tuple>

namespace libtempo::distance {

  namespace internal {

    /// Squared Euclidean Distance dim 1
    template<typename FloatType, typename D>
    [[nodiscard]] inline FloatType sqed1(const D& lines, size_t li, const D& cols, size_t co) {
      const FloatType d =
        lines[li]
        -cols[co];
      return d*d;
    }

    /// Squared Euclidean Distance dim N
    template<typename FloatType, typename D>
    [[nodiscard]] inline FloatType sqedN(const D& lines, size_t li, const D& cols, size_t co, size_t ndim) {
      const size_t li_offset = li*ndim;
      const size_t co_offset = co*ndim;
      FloatType acc{0};
      for (size_t k{0}; k<ndim; ++k) {
        FloatType d = lines[li_offset+k]-cols[co_offset+k];
        acc += d*d;
      }
      return acc;
    }

  } // End of namespace internal

  /// Get the univariate squares Euclidean distance
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sqed(){ return internal::sqed1<FloatType, D>; }

  /// Get the multivariate squared Euclidean distance
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sqed(size_t ndim) {
    return [ndim](const D& lines, size_t li, const D& cols, size_t co) {
      return internal::sqedN<FloatType, D>(lines, li, cols, co, ndim);
    };
  }



  /// Type alias for a tuple representing (nblines, nbcols)
  template<typename D>
  using lico_t = std::tuple<const D&, size_t, const D&, size_t>;

  /// Helper function checking and ordering the length of the series
  template<typename FloatType, typename D>
  [[nodiscard]] inline std::variant<FloatType, lico_t<D>> check_order_series(
      const D& s1, size_t length1,
      const D& s2, size_t length2
    ) {
    // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
    if (length1==0 && length2==0) { return {FloatType(0.0)}; }
    else if ((length1==0)!=(length2==0)) { return utils::PINF<FloatType>; }
    // Use the smallest size as the columns (which will be the allocation size)
    return (length1>length2) ? std::forward_as_tuple(s1, length1, s2, length2) : std::forward_as_tuple(s2, length2, s1, length1);
  }

} // Enf of namespace libtempo::distance
