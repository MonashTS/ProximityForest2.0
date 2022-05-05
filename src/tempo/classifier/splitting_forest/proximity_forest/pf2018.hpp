#pragma once

#include <tempo/predef.hpp>
#include <tempo/utils/utils.hpp>
#include <tempo/tseries/dataset.hpp>

namespace tempo::classifier::pf {


  struct PF2018 {

    /// Constructor with the forest parameters
    PF2018(size_t nb_trees, size_t nb_candidates, std::optional<size_t> opt_seed={});

    /// Train the forest
    void train(DTSMap const& trainset, size_t nb_threads=1);

    /// Prediction
    void predict(DTSMap const& testset);

  private:
    // Configuration
    DTS trainset;
    size_t nb_trees{5};
    size_t nb_candidates{100};
    size_t seed;
    // PIMPL
    struct Impl;
    std::unique_ptr<Impl> pImpl;

  };

} // End of namespace tempo::classifier::pf