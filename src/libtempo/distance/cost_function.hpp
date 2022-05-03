#pragma once

#include <libtempo/utils/utils.hpp>

namespace libtempo::distance {

  namespace univariate {

    /// CFunBuilder Univariate Absolute difference exponent 1
    template<Float F, Subscriptable D>
    auto ad1(const D& lines, const D& cols) {
      return [&](size_t i, size_t j) {
        const F d = lines[i] - cols[j];
        return std::abs(d);
      };
    }

    /// CFunBuilder Univariate Absolute difference exponent 2
    template<Float F, Subscriptable D>
    auto ad2(const D& lines, const D& cols) {
      return [&](size_t i, size_t j) {
        const F d = lines[i] - cols[j];
        return d*d;
      };
    }

    /// CFunBuilder Univariate Absolute difference exponent e
    template<Float F, Subscriptable D>
    auto ade(const F e) {
      return [e](const D& lines, const D& cols) {
        return [&, e](size_t i, size_t j) {
          const F d = std::abs(lines[i] - cols[j]);
          return std::pow(d, e);
        };
      };
    }
  }

  namespace multivariate {

    /// CFunBuilder Multivariate Absolute difference exponent 2
    template<Float F, Subscriptable D>
    auto ad2N(const D& lines, const D& cols, size_t ndim) {
      return [&, ndim](size_t i, size_t j) {
        const size_t li_offset = i*ndim;
        const size_t co_offset = j*ndim;
        F acc{0};
        for (size_t k{0}; k<ndim; ++k) {
          const auto d = lines[li_offset + k] - cols[co_offset + k];
          acc += d*d;
        }
        return acc;
      };
    }
  }

} // Enf of namespace libtempo::distance
