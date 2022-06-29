#pragma once

#include <tempo/dataset/dts.hpp>
#include <tempo/utils/utils.hpp>
#include <tempo/distance/elastic/dtw.hpp>
#include <tempo/distance/lockstep/direct.hpp>
#include <tempo/distance/lockstep/lockstep.hpp>
#include <tempo/transform/normalization.hpp>


/// Alias type for distance functions
using distfun_t = std::function<double(tempo::TSeries const& A, tempo::TSeries const& B, double ub)>;
