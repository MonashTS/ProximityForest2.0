#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/tseries.univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct TWE : public BaseDist {
    F nu;
    F lambda;

    TWE(std::string tname, F nu, F lambda) : BaseDist(std::move(tname)), nu(nu), lambda(lambda) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::twe(t1, t2, nu, lambda, bsf);
    }

    std::string get_distance_name() override { return "TWE:" + std::to_string(nu) + ":" + std::to_string(lambda); }
  };

  struct TWEGen : public i_GenDist {
    TransformGetter get_transform;
    T_GetterState<F> get_nu;
    T_GetterState<F> get_lambda;

    TWEGen(TransformGetter get_transform, T_GetterState<F> get_nu, T_GetterState<F> get_lambda) :
      get_transform(std::move(get_transform)), get_nu(std::move(get_nu)), get_lambda(std::move(get_lambda)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /* d */, const ByClassMap& /* bcm */) override {
      const std::string tn = get_transform(state);
      const F nu = get_nu(state);
      const F lambda = get_lambda(state);
      return std::make_unique<TWE>(tn, nu, lambda);
    }
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
