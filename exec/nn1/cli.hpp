#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/utils/simplecli.hpp>
#include <tempo/dataset/dts.hpp>

#include <algorithm>

extern std::string usage;

using distfun_t = std::function<tempo::F(tempo::TSeries const&, tempo::TSeries const&, tempo::F)>;

/// Info when computing the distance only between two series
struct PairWise {
  std::string src1;
  size_t idx1;
  std::string src2;
  size_t idx2;
};

/// Structure for command line argument
struct Config {
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Optional args
  size_t nbthreads;
  size_t seed;
  std::unique_ptr<tempo::PRNG> pprng;
  std::optional<std::filesystem::path> outpath{};
  std::optional<PairWise> opair{};

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
  std::optional<long> param_window{};             // DTW ERP LCSS
  std::optional<double> param_cf_exponent{};      // All but LCSS MSM and TWE
  std::optional<double> param_omega{};            // ADTW
  std::optional<double> param_gap_value{};        // ERP
  std::optional<double> param_g{};                // WDTW
  std::optional<double> param_epsilon{};          // LCSS
  std::optional<double> param_c{};                // MSM
  std::optional<double> param_lambda{};           // TWE
  std::optional<double> param_nu{};               // TWE

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // To Json
  nlohmann::json to_json() const {
    nlohmann::json jv;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Params
    {
      nlohmann::json j;
      j["seed"] = (long)seed;
      if (outpath) { j["outpath"] = outpath.value().string(); }
      jv["general"] = j;
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Loaded dataset & normalisation per series
    {
      jv["train"] = loaded_train_split.header().to_json();
      jv["test"] = loaded_test_split.header().to_json();
      // Normalisation
      nlohmann::json j;
      j["name"] = normalisation_name;
      if (norm_min_range) { j["min_range"] = norm_min_range.value(); }
      if (norm_max_range) { j["max_range"] = norm_max_range.value(); }
      //
      jv["normalisation"] = std::move(j);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Transform
    {
      nlohmann::json j;
      //
      j["name"] = transform_name;
      if (param_derivative_degree) { j["degree"] = param_derivative_degree.value(); }
      //
      jv["transform"] = std::move(j);
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Distance
    {
      nlohmann::json j;
      //
      j["name"] = dist_name;
      if (param_window) { j["window"] = param_window.value(); }
      if (param_cf_exponent) { j["cf_exponent"] = param_cf_exponent.value(); }
      if (param_omega) { j["omega"] = param_omega.value(); }
      if (param_gap_value) { j["gap_value"] = param_gap_value.value(); }
      if (param_g) { j["g"] = param_g.value(); }
      if (param_epsilon) { j["epsilon"] = param_epsilon.value(); }
      if (param_c) { j["c"] = param_c.value(); }
      if (param_lambda) { j["lambda"] = param_lambda.value(); }
      if (param_nu) { j["nu"] = param_nu.value(); }
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
