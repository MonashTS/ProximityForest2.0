#pragma once

#include "pch.h"
#include <tempo/utils/simplecli.hpp>


extern std::string usage;

/// Structure for command line argument
struct dist_config {

  std::string dist_name;
  distfun_t dist_fun;
  std::optional<long> param_window{};
  std::optional<double> param_cf_exponent{};
  std::optional<double> param_omega{};

  [[nodiscard]] Json::Value to_json() const {
    Json::Value j;
    j["name"] = dist_name;
    if (param_window) { j["window"] = param_window.value(); }
    if (param_cf_exponent) { j["cf_exponent"] = param_cf_exponent.value(); }
    if (param_omega) { j["omega"] = param_omega.value(); }
    return j;
  }

};

/// Exit with a code and a message
[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {});

/// Parse command line argument
dist_config cmd_dist(std::vector<std::string> const& args);
