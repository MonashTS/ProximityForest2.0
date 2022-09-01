#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

namespace tempo::classifier::SForest::splitter::nn1 {

  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Generate a transform name suitable for elastic distances
  template<typename TrainState>
  using TransformGetter = std::function<std::string(TrainState& train_state)>;

  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Generate an cfe e used in some elastic distances' cost function cost(a,b)=|a-b|^e
  template<typename TrainState>
  using ExponentGetter = std::function<double(TrainState& train_state)>;

  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Generate a warping window
  template<typename TrainState, typename TrainData>
  using WindowGetter = std::function<size_t(TrainState& train_state, TrainData const& train_data)>;



  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Some distances (like ADTW, ERP and LCSS) generate a random value based on the dataset
  /// (requires 'data' and the dataset name, and 'bcm' for the local subset)
  template<typename TrainState, typename TrainData>
  using StatGetter = std::function<F(TrainState& state, TrainData const& data,
                                     ByClassMap const& bcm, std::string const& tn)>;

}