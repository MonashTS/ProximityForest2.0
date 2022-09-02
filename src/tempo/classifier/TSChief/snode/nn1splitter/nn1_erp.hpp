#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct ERP : public BaseDist {

    F cfe;
    F gv;
    size_t w;

    ERP(std::string tname, F cfe, F gv, size_t w) : BaseDist(std::move(tname)), cfe(cfe), gv(gv), w(w) {}

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override {
      return distance::univariate::erp(t1.rawdata(), t1.size(), t2.rawdata(), t2.size(), cfe, gv, w, bsf);
    }

    std::string get_distance_name() override {
      return "ERP:" + std::to_string(cfe) + ":" + std::to_string(gv) + ":" + std::to_string(w);
    }
  };

  struct ERPGen : public i_GenDist {

    TransformGetter get_transform;
    ExponentGetter get_cfe;
    StatGetter get_gv;
    WindowGetter get_win;

    ERPGen(TransformGetter get_transform, ExponentGetter get_cfe, StatGetter get_gv, WindowGetter get_win) :
      get_transform(std::move(get_transform)),
      get_cfe(std::move(get_cfe)),
      get_gv(std::move(get_gv)),
      get_win(std::move(get_win)) {}

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) override {
      const std::string tn = get_transform(state);
      const double e = get_cfe(state);
      const double gv = get_gv(state, data, bcm, tn);
      const size_t w = get_win(state, data);
      return std::make_unique<ERP>(tn, e, gv, w);
    }

  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
