#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>

namespace tempo::distance {
  // Minkowski distance is only defined for univariate matrices

  /// Minkowski distance on arma::Row
  template<Float F>
  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p, [[maybe_unused]]F ub = utils::PINF<F>) {
    return std::pow(arma::sum(arma::pow(arma::abs(A - B), p)), (F)1.0/p);
  }

  /// Minkowski distance on TSeries
  inline F minkowski(TSeries const& A, TSeries const& B, F p, F ub = utils::PINF<F>) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return minkowski<F>(a, b, p, ub);
  }

} // End of namespace::distance