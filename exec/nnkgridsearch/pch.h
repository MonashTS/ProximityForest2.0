#pragma once

#include <algorithm>

#include <tempo/utils/utils.hpp>

#include <tempo/dataset/dts.hpp>
#include <tempo/dataset/dataset.hpp>
#include <tempo/dataset/tseries.hpp>

/// Alias type for distance functions
using distfun_t = std::function<double(tempo::TSeries const& A, tempo::TSeries const& B, double ub)>;
