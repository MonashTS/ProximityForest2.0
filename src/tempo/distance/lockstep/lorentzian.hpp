#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::distance {

  /// Lorentzian metric on arma row vector
  F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B) {
    return arma::sum(arma::log(1 + arma::abs(A - B)));
  }

  /// Lorentzian metric on TSeries (univariate only)
  inline F lorentzian(TSeries const& A, TSeries const& B) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return lorentzian(a, b);
  }

} // End of namespace::distance
