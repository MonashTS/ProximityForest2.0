#pragma once

#include <libtempo/tseries/tseries.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>

namespace libtempo::transform {

  /** Computation of a series derivative according to "Derivative Dynamic Time Warping" by Keogh & Pazzani
   * @tparam T            Input series, must be in
   * @param series        Pointer to the series's data
   * @param length        Length of the series
   * @param out           Pointer where to write the derivative. Must be able to store 'length' values.
   * Warning: series and out should not overlap (i.e. not in place derivation)
   */
  template<Float F, std::random_access_iterator Input, std::output_iterator<F> Output>
  void derivative(const Input& series, size_t length, Output out) {
    if (length>2) {
      for (size_t i{1}; i<length-1; ++i) {
        out[i] = ((series[i]-series[i-1])+((series[i+1]-series[i-1])/2.0))/2.0;
      }
      out[0] = out[1];
      out[length-1] = out[length-2];
    } else {
      std::copy(series, series+length, out);
    }
  }

  template<Float F, Label L>
  [[nodiscard]] inline std::vector<DTS<F, L>>
  derive(const DTS<F, L>& input, size_t degree) {
    assert(degree>0);
    assert(input.core().nb_dimensions() == 1);

    std::vector<DTS<F, L>> result;

    // 1st derivative
    {
      std::vector<TSeries<F, L>> series;
      for (size_t i = 0; i<input.size(); ++i) {
        const auto& ts = input[i];
        const size_t l = ts.length();
        std::vector<F> d(l);
        derivative<F>(ts.rm_data(), l, d.data());
        series.template emplace_back(std::move(TSeries<F,L>::mk_rowmajor(ts, std::move(d))));
      }

      result.template emplace_back(input, "derivative", std::move(series));
    }

    return result;
  }


}