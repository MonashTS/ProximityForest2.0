#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>

namespace tempo::distance {

  /// Lorentzian metric on arma vector (Row or Col)
  template<Float F, typename ARMA_V>
  F lorentzian(ARMA_V const& A, ARMA_V const& B, [[maybe_unused]]F ub = utils::PINF<F>) {
    return arma::sum(arma::log(1 + arma::abs(A - B)));
  }

  /// Lorentzian metric on TSeries (univariate only)
  inline F lorentzian(TSeries const& A, TSeries const& B, F ub = utils::PINF<F>) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return lorentzian<F>(a, b, ub);
  }

} // End of namespace::distance
