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
      const FloatType d = lines[li]-cols[co];
      return d*d;
    }

    /// Squared Euclidean Distance dim 1
    template<typename FloatType, typename D>
    [[nodiscard]] inline FloatType ed(const D& lines, size_t li, const D& cols, size_t co) {
      return std::sqrt(sqed1(lines, li, cols, co));
    }

    /// Squared Euclidean Distance dim N
    template<typename FloatType, typename D>
    [[nodiscard]] inline FloatType sqedN(const D& lines, size_t li, const D& cols, size_t co, size_t ndim) {
      const size_t li_offset = li*ndim;
      const size_t co_offset = co*ndim;
      FloatType acc{0};
      for (size_t k{0}; k<ndim; ++k) {
        const FloatType d = lines[li_offset+k]-cols[co_offset+k];
        acc += d*d;
      }
      return acc;
    }

    /// Euclidean Distance dim N
    template<typename FloatType, typename D>
    [[nodiscard]] inline FloatType edN(const D& lines, size_t li, const D& cols, size_t co, size_t ndim) {
      return std::sqrt(sqedN<FloatType, D>(lines, li, cols, co, ndim));
    }

    /// Euclidean Distance to midpoint dim N - used by MSM
    template<typename FloatType, typename D>
    [[nodiscard]] inline FloatType edNmid(const D& X, size_t xnew, size_t xi, const D& Y, size_t yi, size_t ndim) {
      const size_t xnew_offset = xnew*ndim;
      const size_t xi_offset = xi*ndim;
      const size_t yi_offset = yi*ndim;
      FloatType acc{0};
      for (size_t k{0}; k<ndim; ++k) {
        const FloatType mid = (X[xi_offset+k]+Y[yi_offset+k])/2;
        const FloatType dmid = mid-X[xnew_offset+k];
        acc += dmid*dmid;
      }
      return std::sqrt(acc);
    }

  } // End of namespace internal


  /// Get the univariate absolute distance
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto abs() {
    return [](const D& lines, size_t li, const D& cols, size_t co) { return std::abs(lines[li]-cols[co]); };
  }

  /// Get the univariate squares Euclidean distance
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto ed() { return internal::ed<FloatType, D>; }

  /// Get the univariate squares Euclidean distance
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sqed() { return internal::sqed1<FloatType, D>; }

  /// Get the multivariate Euclidean distance
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto ed(size_t ndim) {
    return [ndim](const D& lines, size_t li, const D& cols, size_t co) {
      return internal::edN<FloatType, D>(lines, li, cols, co, ndim);
    };
  }

  /// Get the multivariate squared Euclidean distance
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto sqed(size_t ndim) {
    return [ndim](const D& lines, size_t li, const D& cols, size_t co) {
      return internal::sqedN<FloatType, D>(lines, li, cols, co, ndim);
    };
  }

  /// Get the Euclidean distance between xnew and the midpoint of xi and yi - Use by Multivariate MSM cost
  template<typename FloatType, typename D>
  [[nodiscard]] inline auto edNmid(size_t ndim) {
    return [ndim](const D& X, size_t xnew, size_t xi, const D& Y, size_t yi) {
      return internal::edNmid<FloatType, D>(X, xnew, xi, Y, yi, ndim);
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
    return (length1>length2) ? std::forward_as_tuple(s1, length1, s2, length2) : std::forward_as_tuple(s2, length2, s1,
      length1);
  }

} // Enf of namespace libtempo::distance
