#pragma once

#include "pch.h"
#include <tempo/utils/simplecli.hpp>

extern std::string usage;

/// Structure for command line argument
struct Config {
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Optional args
  size_t k;
  size_t nbthreads;
  size_t seed;
  std::unique_ptr<tempo::PRNG> pprng;
  std::optional<std::filesystem::path> outpath{};

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Loaded dataset + normalisation
  tempo::DTS loaded_train_split;
  tempo::DTS loaded_test_split;
  std::string normalisation_name{"null"};
  std::optional<double> norm_min_range;
  std::optional<double> norm_max_range;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // TRANSFORM
  tempo::DTS train_split;
  tempo::DTS test_split;
  std::string transform_name{"null"};
  std::optional<int> param_derivative_degree;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DISTANCE

  std::string dist_name;
  distfun_t dist_fun;
  std::optional<long> param_window{};
  std::optional<double> param_cf_exponent{};
  std::optional<double> param_omega{};
  std::optional<double> param_gap_value{};

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // To Json
  [[nodiscard]] Json::Value to_json() const {
    Json::Value jv;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Params
    {
      Json::Value j;
      j["k"] = (int)k;
      j["seed"] = (long)seed;
      if (outpath) { j["outpath"] = outpath.value().string(); }
      jv["general"] = j;
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Loaded dataset & normalisation per series
    {
      jv["train"] = train_split.header().to_json();
      jv["test"] = test_split.header().to_json();
      // Normalisation
      Json::Value j;
      j["name"] = normalisation_name;
      if (norm_min_range) { j["min_range"] = norm_min_range.value(); }
      if (norm_max_range) { j["max_range"] = norm_max_range.value(); }
      //
      jv["normalisation"] = std::move(j);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Transform
    {
      Json::Value j;
      //
      j["name"] = transform_name;
      if (param_derivative_degree) { j["degree"] = param_derivative_degree.value(); }
      //
      jv["transform"] = std::move(j);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Distance
    {
      Json::Value j;
      //
      j["name"] = dist_name;
      if (param_window) { j["window"] = param_window.value(); }
      if (param_cf_exponent) { j["cf_exponent"] = param_cf_exponent.value(); }
      if (param_omega) { j["omega"] = param_omega.value(); }
      if (param_gap_value) { j["gap_value"] = param_gap_value.value(); }
      //
      jv["distance"] = std::move(j);
    }

    return jv;
  }
};

/// Exit with a code and a message
[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {});

/// Read/init optional args from the command line
void cmd_optional(std::vector<std::string> const& args, Config& conf);

/// Updated config parsing the normalisation-related args
void cmd_normalisation(std::vector<std::string> const& args, Config& conf);

/// Updated config parsing the transform-related args
void cmd_transform(std::vector<std::string> const& args, Config& conf);

/// Updated config parsing the distance-related args
void cmd_dist(std::vector<std::string> const& args, Config& conf);
