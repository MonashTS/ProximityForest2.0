#pragma once

#include <algorithm>

#include <tempo/utils/utils.hpp>

#include <tempo/dataset/dts.hpp>
#include <tempo/dataset/dataset.hpp>
#include <tempo/dataset/tseries.hpp>

#include "tempo/transform/derivative.hpp"
#include <tempo/transform/normalization.hpp>

#include <tempo/distance/lockstep/direct.hpp>
#include <tempo/distance/lockstep/lockstep.hpp>

#include <tempo/distance/sliding/cross_correlation.hpp>

#include <tempo/distance/elastic/dtw.hpp>
#include "tempo/distance/elastic/adtw.hpp"
#include "tempo/distance/elastic/erp.hpp"


/// Alias type for distance functions
using distfun_t = std::function<double(tempo::TSeries const& A, tempo::TSeries const& B, double ub)>;
