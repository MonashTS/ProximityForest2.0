#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>

namespace tempo::distance {

  /// Manhattan metric on arma vector (Row or Col)
  template<Float F, typename ARMA_V>
  F manhattan(ARMA_V const& A, ARMA_V const& B, F p, [[maybe_unused]]F ub = utils::PINF<F>) {
    return arma::sum(arma::abs(A - B));
  }

  /// manhattan metric on TSeries (univariate only)
  inline F manhattan(TSeries const& A, TSeries const& B, F p, F ub = utils::PINF<F>) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return manhattan<F>(a, b, p, ub);
  }

}  // End of namespace tempo::distance