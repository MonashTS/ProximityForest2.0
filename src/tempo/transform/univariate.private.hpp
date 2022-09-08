#pragma once

#include <cstddef>

#include "core/univariate.derivative.hpp"
#include "core/univariate.normalization.hpp"

namespace tempo::transform::univariate {

  // --- --- --- Derivative

  template<typename F>
  void derive(F const* data, size_t length, F* output){
    tempo::transform::core::univariate::derive<F, F const*, F*>(data, length, output);
  }

  // --- --- --- Normalisation

  template<typename F>
  void minmax(F const* data, size_t length, F* output, F range_min, F range_max){
    tempo::transform::core::univariate::minmax<F>(data, length, output, range_min, range_max);
  }

  template<typename F>
  arma::Row<F> minmax(arma::Row<F> const& A, F range_min, F range_max) {
    return tempo::transform::core::univariate::minmax<F>(A, range_min, range_max);
  }

  template<typename F>
  void percentile_minmax(F const *data, size_t length, F *output, size_t p, F range_min, F range_max) {
    tempo::transform::core::univariate::percentile_minmax<F>(data, length, output, p, range_min, range_max);
  }

  template<typename F>
  arma::Row<F> percentile_minmax(arma::Row<F> const& A, size_t p, F range_min, F range_max) {
    return tempo::transform::core::univariate::percentile_minmax(A, p, range_min, range_max);
  }

  template<typename F>
  void meannorm(F const *data, size_t length, F *output){
    tempo::transform::core::univariate::meannorm(data, length, output);
  }

  template<typename F>
  arma::Row<F> meannorm(arma::Row<F> const& A) {
    return tempo::transform::core::univariate::meannorm(A);
  }

  template<typename F>
  void unitlength(F const *data, size_t length, F *output){
    tempo::transform::core::univariate::unitlength(data, length, output);
  }

  template<typename F>
  arma::Row<F> unitlength(arma::Row<F> const& A) {
    return tempo::transform::core::univariate::unitlength(A);
  }

  template<typename F>
  void zscore(F const *data, size_t length, F *output){
    tempo::transform::core::univariate::zscore(data, length, output);
  }

  template<typename F>
  arma::Row<F> zscore(arma::Row<F> const& A) {
    return tempo::transform::core::univariate::zscore(A);
  }

} // End of namespace tempo::transform::univariate
