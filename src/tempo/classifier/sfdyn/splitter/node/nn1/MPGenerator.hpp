#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/sfdyn/stree.hpp>

namespace tempo::classifier::sf::node::nn1dist {

  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Generate a transform name suitable for elastic distances
  using TransformGetter = std::function<std::string(TreeState& state)>;

  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Generate an exponent e used in some elastic distances' cost function cost(a,b)=|a-b|^e
  using ExponentGetter = std::function<double(TreeState& state)>;

  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Generate a warping window
  using WindowGetter = std::function<size_t(TreeState& state, TreeData const& data)>;

  // --- --- --- --- --- --- --- --- --- --- --- ---

  /// Some distances (like ADTW, ERP and LCSS) generate a random value based on the dataset
  /// (requires 'data' and the dataset name, and 'bcm' for the local subset)
  using StatGetter = std::function<F(TreeState& state, TreeData const& data, ByClassMap const& bcm,
                                     std::string const& tn)>;

}