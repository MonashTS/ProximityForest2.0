#pragma once

#include "pch_lib.hpp"

namespace tempo {

  using LabelType = std::string;
  using L = LabelType;

  using EncodedLabelType = size_t;
  using EL = EncodedLabelType;

  using FloatType = double;
  using F = FloatType;

  using PRNG = std::mt19937_64;

} // End of namespace tempo