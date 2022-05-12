#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>

namespace tempo::distance {

  /// Manhattan metric on arma row vector
  template<Float F>
  F manhattan(arma::Row<F> const& A, arma::Row<F> const& B) {
    return arma::sum(arma::abs(A - B));
  }

  /// Manhattan metric on TSeries (univariate only)
  inline F manhattan(TSeries const& A, TSeries const& B) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return manhattan<F>(a, b);
  }

}  // End of namespace tempo::distance