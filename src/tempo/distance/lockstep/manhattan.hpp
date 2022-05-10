#pragma once

#include "minkowski.hpp"

namespace tempo::distance {

  /// Manhattan metric on arma::Row
  template<Float F>
  F manhattan(arma::Row<F> const& A, arma::Row<F> const& B, F p, [[maybe_unused]]F ub = utils::PINF<F>) {
    return minkowski<F>(A, B, 1, ub);
  }

  /// manhattan metric on TSeries (univariate only)
  inline F manhattan(TSeries const& A, TSeries const& B, F p, F ub = utils::PINF<F>) {
    arma::Row<F> a = A.rowvec();
    arma::Row<F> b = B.rowvec();
    return manhattan<F>(a, b, p, ub);
  }

}  // End of namespace tempo::distance