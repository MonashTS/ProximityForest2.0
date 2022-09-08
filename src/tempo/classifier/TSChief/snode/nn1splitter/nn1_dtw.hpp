#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/tseries.univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct DTW : public BaseDist {
    F cfe;
    size_t w;

    DTW(std::string tname, F cfe, size_t w) : BaseDist(std::move(tname)), cfe(cfe), w(w){}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::dtw(t1, t2, cfe, w, bsf);
    }

    std::string get_distance_name() override { return "DTW:" + std::to_string(cfe) + ":" + std::to_string(w); }
  };

  struct DTWGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_cfe;
    WindowGetter get_win;

    DTWGen(TransformGetter gt, ExponentGetter get_cfe, WindowGetter get_win) :
      get_transform(std::move(gt)), get_cfe(std::move(get_cfe)), get_win(std::move(get_win)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& /* bcm */) override {
      const std::string tn = get_transform(state);
      const F e = get_cfe(state);
      const size_t w = get_win(state, data);
      return std::make_unique<DTW>(tn, e, w);
    }
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter