#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct MSM : public BaseDist {
    F cost;

    MSM(std::string tname, F cost) : BaseDist(std::move(tname)), cost(cost) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::msm(t1.rawdata(), t1.size(), t2.rawdata(), t2.size(), cost, bsf);
    }

    std::string get_distance_name() override { return "MSM:" + std::to_string(cost); }
  };

  struct MSMGen : public i_GenDist {
    TransformGetter get_transform;
    T_GetterState<F> get_cost;

    MSMGen(TransformGetter get_transform, T_GetterState<F> get_cost) :
      get_transform(std::move(get_transform)), get_cost(std::move(get_cost)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /* d */, const ByClassMap& /* bcm */) override {
      const std::string tn = get_transform(state);
      const F cost = get_cost(state);
      return std::make_unique<MSM>(tn, cost);
    }
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
