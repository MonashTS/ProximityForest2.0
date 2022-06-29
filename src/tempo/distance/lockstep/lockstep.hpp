#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>

namespace tempo::distance {

  /// Lorentzian metric on arma row vector
  F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B);

  /// Lorentzian metric on TSeries (univariate only)
  F lorentzian(TSeries const& A, TSeries const& B);

  /// Minkowski metric on arma row vector
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p);

  /// Minkowski metric on TSeries (univariate only)
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  F minkowski(TSeries const& A, TSeries const& B, F p);

  /// Manhattan metric on arma row vector
  F manhattan(arma::Row<F> const& A, arma::Row<F> const& B);

  /// Manhattan metric on TSeries (univariate only)
  F manhattan(TSeries const& A, TSeries const& B);

} // End of namespace::distance
