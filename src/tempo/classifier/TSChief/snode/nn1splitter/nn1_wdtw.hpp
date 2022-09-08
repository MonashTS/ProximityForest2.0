#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/tseries.univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct WDTW : public BaseDist {
    F cfe;
    F g;
    std::vector<F> weights;

    WDTW(std::string tname, F cfe, F g, std::vector<F>&& weights) :
      BaseDist(std::move(tname)), cfe(cfe), g(g), weights(std::move(weights)) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::wdtw(t1, t2, cfe, weights.data(), bsf);
    }

    std::string get_distance_name() override { return "WDTW:" + std::to_string(cfe) + ":" + std::to_string(g); }
  };

  struct WDTWGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_cfe;
    size_t maxl;

    WDTWGen(TransformGetter gt, ExponentGetter get_cfe, size_t maxl) :
      get_transform(std::move(gt)), get_cfe(std::move(get_cfe)), maxl(maxl) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /* data */, const ByClassMap& /* bcm */) override {
      const std::string tn = get_transform(state);
      const F cfe = get_cfe(state);
      const F g = std::uniform_real_distribution<F>(0, 1)(state.prng);
      return std::make_unique<WDTW>(tn, cfe, g, distance::univariate::wdtw_weights(g, maxl));
    }
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
