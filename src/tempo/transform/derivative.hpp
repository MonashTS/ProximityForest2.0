#pragma once

#include "tempo/utils/utils.hpp"
#include "tempo/tseries/tseries.hpp"
#include "tempo/tseries/dataset.hpp"

namespace tempo::transform {

  /** Computation of a series derivative according to "Derivative Dynamic Time Warping" by Keogh & Pazzani
   * @tparam T            Input series, must be in
   * @param series        Pointer to the series's data
   * @param length        Length of the series
   * @param out           Pointer where to write the derivative. Must be able to store 'length' values.
   * Warning: series and out should not overlap (i.e. not in place derivation)
   */
  template<Float F, std::random_access_iterator Input, std::output_iterator<F> Output>
  void derive(const Input& series, size_t length, Output out) {
    if (length>2) {
      for (size_t i{1}; i<length - 1; ++i) {
        out[i] = ((series[i] - series[i - 1]) + ((series[i + 1] - series[i - 1])/2.0))/2.0;
      }
      out[0] = out[1];
      out[length - 1] = out[length - 2];
    } else {
      std::copy(series, series + length, out);
    }
  }

  /** Derive all the series in the 'input' dataset, up to 'degree'.
   *  Returns a vector of 'degree" datasets, where the first one correspond to the 1st derivative,
   *  the 2nd one to the second derivative, etc...
   *
   * @tparam F Floating point type
   * @param input Input dataset - must be univariate
   * @param degree maximum desired derivative >= 1
   */
  [[nodiscard]] inline std::vector<DTS> derive(const DTS& input, size_t degree) {
    assert(degree>0);
    assert(input.header().nb_dimensions()==1);

    std::vector<DTS> result;

    // 1st derivative
    {
      std::vector<TSeries> series;
      for (size_t i = 0; i<input.size(); ++i) {
        const auto& ts = input[i];
        const size_t l = ts.length();
        std::vector<F> d(l);
        derive<F>(ts.rawdata(), l, d.data());
        series.push_back(TSeries::mk_rowmajor(ts, std::move(d)));
      }

      result.emplace_back(input, "derivative", std::move(series), std::optional(Json::Value((int)1)));
    }

    // Following derivative
    for (size_t deg = 2; deg<=degree; ++deg) {

      std::vector<TSeries> series;
      for (size_t i = 0; i<input.size(); ++i) {
        const auto& ts = result.back()[i];
        const size_t l = ts.length();
        std::vector<F> d(l);
        derive<F>(ts.rawdata(), l, d.data());
        series.push_back(TSeries::mk_rowmajor(ts, std::move(d)));
      }

      result.emplace_back(input, "derivative", std::move(series), std::optional(Json::Value((int)deg)));
    }

    return result;
  }

}