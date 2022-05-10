#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>

namespace tempo::distance {

  /// Lorentzian metric on arma::Row
  // d = sum(log(1 + abs(P - Q)));
  template<Float F>
  F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B, [[maybe_unused]]F ub = utils::PINF<F>) {
    return arma::sum(arma::log(1 + arma::abs(A - B)));
  }

  /// Lorentzian metric on TSeries (univariate only)
  inline F lorentzian(TSeries const& A, TSeries const& B, F ub = utils::PINF<F>) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return lorentzian<F>(a, b, ub);
  }

} // End of namespace::distance
