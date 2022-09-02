#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct DTW : public BaseDist {

    double cfe;
    size_t w;

    DTW(std::string tname, double cfe, size_t w) : BaseDist(std::move(tname)), cfe(cfe), w(w){}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::dtw(t1.rawdata(), t1.size(), t2.rawdata(), t2.size(), cfe, w, bsf);
    }

    std::string get_distance_name() override { return "DTW:" + std::to_string(cfe) + ":" + std::to_string(w); }
  };

  struct DTWGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_fce;
    WindowGetter get_win;

    DTWGen(TransformGetter gt, ExponentGetter get_cfe, WindowGetter get_win) :
      get_transform(std::move(gt)), get_fce(std::move(get_cfe)), get_win(std::move(get_win)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& /* bcm */) override {
      // Generate args
      std::string tn = get_transform(state);
      size_t w = get_win(state, data);
      double e = get_fce(state);
      // Build return
      return std::make_unique<DTW>(tn, e, w);
    }
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
