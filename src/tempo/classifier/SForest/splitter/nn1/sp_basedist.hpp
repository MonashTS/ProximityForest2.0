#pragma once

#include <utility>

#include "nn1splitters.hpp"

namespace tempo::classifier::SForest::splitter::nn1 {

  /// Implement some common behaviours for distances
  struct BaseDist_i : public Distance_i {

    /// Store the name of the transform
    std::string transformation_name;

    /// Name of the transformation to draw the data from
    std::string get_transformation_name() override{ return transformation_name; }

    // ---

    explicit BaseDist_i(std::string str): transformation_name(std::move(str)){ }

    virtual ~BaseDist_i() = default;
  };

}
