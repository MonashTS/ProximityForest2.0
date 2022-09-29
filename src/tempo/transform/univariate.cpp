#include "univariate.private.hpp"

namespace tempo::transform::univariate {

  // Implementation through template explicit instantiation

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Double implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  using F = double;

  // --- --- --- Derivative

  template void derive<F>(F const* data, size_t length, F* output);
  template void derive<F>(F const* data, size_t length, F* output, size_t degree);

  // --- --- --- Normalisation

  template void minmax<F>(F const* data, size_t length, F* output, F rmin, F rmax);

  template void percentile_minmax(F const *data, size_t length, F *output, size_t p, F range_min, F range_max);

  template void meannorm(F const *data, size_t length, F *output);

  template void unitlength(F const *data, size_t length, F *output);

  template void zscore(F const *data, size_t length, F *output);

  // --- --- --- --- --- --- Extra: Armadillo Row Vector

  template arma::Row<F> minmax(arma::Row<F> const& A, F range_min, F range_max);

  template arma::Row<F> percentile_minmax(arma::Row<F> const& A, size_t p, F range_min, F range_max);

  template arma::Row<F> meannorm(arma::Row<F> const& A);

  template arma::Row<F> unitlength(arma::Row<F> const& A);

  template arma::Row<F> zscore(arma::Row<F> const& A);





  /*
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Float implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  using Ff = float;

  template void derive<Ff>(Ff const* data, size_t length, Ff* output);

  */

} // End of namespace tempo::transform::univariate
