#pragma once

#include <tempo/predef.hpp>
#include <tempo/utils/utils.hpp>
#include <tempo/tseries/dataset.hpp>

namespace tempo::classifier {

  struct PF2018 {

    /// Constructor with the forest parameters
    PF2018(size_t nb_trees, size_t nb_candidates);

    // Note: must be non-default for pImpl with std::unique_ptr.
    // Else, the compiler may try to define the destructor here, where the info about Impl are not known.
    ~PF2018();

    /// Train the forest
    /// Note: Make a copy of the DTSMap trainset - copying a DTS is cheap as it is an indirection to the actual data.
    void train(DTSMap trainset, size_t seed, size_t nb_threads);

    /// Probability prediction
    /// Output results in 'out_probabilities' and 'out_weights'
    /// 'out_probabilities' will be a (nb train classes x nb test queries) matrix where
    ///   - each column represents the probabilities per class, as given by the header of the trainset (see 'train')
    ///   - each cell of a column is a probabilities between 0 and 1. A column sums to 1.
    /// 'out_weights' will be a (nb test queries) row vector, where each cell represents the number of train exemplar
    ///   that contributed to the prediction. As this is done over the forest, this number can be higher than the actual
    ///   number of train exemplar.
    void predict(DTSMap const& testset, arma::mat& out_probabilities, arma::rowvec& out_weights,
                 size_t seed, size_t nb_threads);

  private:
    // PIMPL
    struct Impl;
    std::unique_ptr<Impl> pImpl;

  };

} // End of namespace tempo::classifier