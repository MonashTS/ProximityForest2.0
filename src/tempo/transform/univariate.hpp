#pragma once

#include <cstddef>

namespace tempo::transform::univariate {

  // --- --- --- Derivative

  template<typename F>
  void derive(F const* data, size_t length, F* output);

  // --- --- --- Normalisation

  template<typename F>
  void minmax(F const* data, size_t length, F* output, F arg_min, F arg_max);

  template<typename F>
  arma::Row<F> minmax(arma::Row<F> const& A, F range_min, F range_max);

  template<typename F>
  void percentile_minmax(F const *data, size_t length, F *output, size_t p, F range_min, F range_max);

  template<typename F>
  arma::Row<F> percentile_minmax(arma::Row<F> const& A, size_t p, F range_min, F range_max);

  template<typename F>
  void meannorm(F const *data, size_t length, F *output);

  template<typename F>
  arma::Row<F> meannorm(arma::Row<F> const& A);

  template<typename F>
  void unitlength(F const *data, size_t length, F *output);

  template<typename F>
  arma::Row<F> unitlength(arma::Row<F> const& A);

  template<typename F>
  void zscore(F const *data, size_t length, F *output);

  template<typename F>
  arma::Row<F> zscore(arma::Row<F> const& A);

} // End of namespace tempo::transform::univariate
