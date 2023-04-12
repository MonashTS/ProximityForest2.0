#pragma once

#include <armadillo>

namespace tempo::distance::core::univariate {

  /// Lorentzian metric on arma::Row
  template<typename F>
  F lorentzian(arma::Row<F> const& A, arma::Row<F> const& B) {
    return arma::sum(arma::log(1 + arma::abs(A - B)));
  }

  template<typename F>
  F lorentzian(F const *A, size_t lA, F const *B, size_t lB) {
    const arma::Row<F> ra(const_cast<F *>(A), lA, false, true);
    const arma::Row<F> rb(const_cast<F *>(B), lB, false, true);
    return lorentzian<F>(ra, rb);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Minkowski metric on arma::Row
  ///  - Equal to the Manhattan distance with p=1
  ///  - Equal to the Euclidean Distance distance with p=2
  ///  - Also see the direct alignment function, which does the same without taking the root.
  ///    With exponent 0.5, 1, and 2, the direct alignment uses specialised cost function,
  ///    which may be faster, in particular for NN search.
  template<typename F>
  F minkowski(arma::Row<F> const& A, arma::Row<F> const& B, F p) {
    return std::pow(arma::sum(arma::pow(arma::abs(A - B), p)), (F)1.0/p);
  }

  template<typename F>
  F minkowski(F const *A, size_t lA, F const *B, size_t lB, F p) {
    const arma::Row<F> ra(const_cast<F *>(A), lA, false, true);
    const arma::Row<F> rb(const_cast<F *>(B), lB, false, true);
    return minkowski<F>(ra, rb, p);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Manhattan metric on arma::Row - special case of Minkowski
  template<typename F>
  F manhattan(arma::Row<F> const& A, arma::Row<F> const& B) {
    return arma::sum(arma::abs(A - B));
  }

  template<typename F>
  F manhattan(F const *A, size_t lA, F const *B, size_t lB) {
    const arma::Row<F> ra(const_cast<F *>(A), lA, false, true);
    const arma::Row<F> rb(const_cast<F *>(B), lB, false, true);
    return manhattan<F>(ra, rb);
  }

} // End of namespace tempo::distance::core::univariate