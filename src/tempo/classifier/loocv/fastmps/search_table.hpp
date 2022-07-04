#pragma once

#include <tempo/utils/utils.hpp>

namespace tempo::classifier::speedy {

  /// One cell of the table
  struct CandidateNN {

    enum Status {
      /// This is the Nearest Neighbour
      NN,
      /// Best Candidate so far
      BC,
    };

    /// Index of the sequence in train
    size_t nn_index;

    /// Window validity
    size_t r;

    size_t paramValidity;

    /// Computed lower bound
    tempo::F distance;

    Status status;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    CandidateNN() :
      nn_index{std::numeric_limits<size_t>::max()},
      r{std::numeric_limits<size_t>::max()},
      paramValidity{std::numeric_limits<size_t>::max()},
      distance{tempo::utils::PINF},
      status{BC} {}

    CandidateNN(const CandidateNN& other) = default;

    CandidateNN(CandidateNN&& other) = default;

    CandidateNN& operator=(const CandidateNN& other) = default;

    CandidateNN& operator=(CandidateNN&& other) = default;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    //
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  };

} // End of namespace classifier