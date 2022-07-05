#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/distance/helpers.hpp>

namespace tempo::classifier::loocv {

  struct I_DistOptimised {

    /// Act as an optimised distance between two series
    virtual F operator()(TSeries const& T1, TSeries const& T2, F ub) = 0;

    /// JSON output of the optimised distance
    virtual Json::Value to_json() = 0;

  };

  struct I_DistMParams {

    /// Given a parameter index p, generate a distance dist(T1, T2, ub)
    virtual tempo::distance::distfun_t get_distance(size_t p) = 0;

    ///

  };




} // End of namespace tempo::classifier::loocv
