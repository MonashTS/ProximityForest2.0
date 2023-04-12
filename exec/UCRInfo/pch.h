#pragma once

#include <tempo/dataset/dts.hpp>
#include <tempo/utils/utils.hpp>


/// Alias type for distance functions
using distfun_t = std::function<double(tempo::TSeries const& A, tempo::TSeries const& B, double ub)>;
