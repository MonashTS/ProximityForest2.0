#pragma once

#include <algorithm>
#include <concepts>
#include <random>

namespace tempo::transform::core::univariate {

  template<std::floating_point F, std::random_access_iterator Input, std::uniform_random_bit_generator PRNG, std::output_iterator<F> Output>
  void noise(Input const& series, size_t length, F stddev, F delta, PRNG& prng, Output& out) {
   std::normal_distribution<> d{0,stddev};
    for (size_t i=0; i<length; ++i) {
      auto s = delta*d(prng);
      out[i] = series[i] + s;
    }
  }

}