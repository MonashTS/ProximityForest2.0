#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct DTWFull : public BaseDist {
    F cfe;

    DTWFull(std::string tname, F cfe) : BaseDist(std::move(tname)), cfe(cfe) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::dtw(t1.rawdata(), t1.size(), t2.rawdata(), t2.size(), cfe, utils::NO_WINDOW, bsf);
    }

    std::string get_distance_name() override { return "DTWFull:" + std::to_string(cfe); }
  };

  struct DTWFullGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_fce;

    DTWFullGen(TransformGetter gt, ExponentGetter get_cfe) :
      get_transform(std::move(gt)), get_fce(std::move(get_cfe)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /* bcm */) override {
      const std::string tn = get_transform(state);
      const F e = get_fce(state);
      return std::make_unique<DTWFull>(tn, e);
    }
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
