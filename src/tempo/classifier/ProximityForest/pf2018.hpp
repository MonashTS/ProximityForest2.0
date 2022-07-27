#pragma once

#include <tempo/classifier/utils.hpp>
#include <tempo/dataset/dts.hpp>

namespace tempo::classifier {

  struct PF2018 {

    inline static const std::string pf2018_11{"pf2018_11"};
    inline static const std::string pf2018_11_vcfe{"pf2018_11_vcfe"};
    inline static const std::string pf2018_22{"pf2018_22"};
    inline static const std::string pf2018_adtw_lcss{"pf2018_adtw_lcss"};
    inline static const std::set<std::string> valid_versions{pf2018_11, pf2018_11_vcfe, pf2018_22, pf2018_adtw_lcss};

    /// Constructor with the forest parameters
    PF2018(size_t nb_trees, size_t nb_candidates, std::string pfversion);

    // Note: must be non-default for pImpl with std::unique_ptr.
    // Else, the compiler may try to define the destructor here, where the info about Impl are not known.
    ~PF2018();

    /// Train the forest
    /// Note: Make a copy of the DTSMap trainset - copying a DTS is cheap as it is an indirection to the actual data.
    void train(std::map<std::string, DTS> trainset, size_t seed, size_t nb_threads);

    /// Test with the forest
    classifier::ResultN predict(std::map<std::string, DTS> const& testset, size_t seed, size_t nb_threads);

  private:
    // PIMPL
    struct Impl;
    std::unique_ptr<Impl> pImpl;

  };

} // End of namespace tempo::classifier
