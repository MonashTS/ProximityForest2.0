#include "pch.h"
#include <tempo/reader/new_reader.hpp>

#include "cli.hpp"

namespace fs = std::filesystem;

// Given a UCR dataset in ts format, compute some information

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;
  using namespace tempo::utils;

  // --- --- --- --- --- ---
  // Program Arguments
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if (args.empty()) { do_exit(0, usage); }

  // --- --- ---
  // Optional

  // Random seed for sampling
  size_t seed;
  {
    auto p_seed = tempo::scli::get_parameter<long long>(args, "-seed", tempo::scli::extract_int, 0);
    if (p_seed<0) { seed = std::random_device()(); } else { seed = p_seed; }
  }
  tempo::PRNG prng(seed);

  // Output file
  optional<fs::path> outpath{};
  {
    auto p_out = tempo::scli::get_parameter<string>(args, "-out", tempo::scli::extract_string);
    if (p_out) { outpath = {fs::path{p_out.value()}}; }
  }

  // Sample with Modified Minkowsky
  size_t mm_nbsample{0};
  double mm_exponent{0.0};
  {
    auto parg_dist = tempo::scli::get_parameter<string>(args, "-modminkowski", tempo::scli::extract_string);
    if (parg_dist) {
      auto v = tempo::reader::split(parg_dist.value(), ':');
      bool ok = v.size()==2;
      if (ok) {
        auto on = tempo::reader::as_int(v[0]);
        auto oe = tempo::reader::as_double(v[1]);
        ok = oe.has_value()&&oe.has_value();
        if (ok) {
          mm_nbsample = on.value();
          mm_exponent = oe.value();
        }
      } else { do_exit(1, "specify <n>:<e> after '-modminkowski'"); }
    }
  }

  // --- --- ---
  // Mandatory

  // Path to UCR
  fs::path UCRPATH;
  {
    auto parg_UCRPATH = tempo::scli::get_parameter<string>(args, "-p", tempo::scli::extract_string);
    if (!parg_UCRPATH) { do_exit(1, "specify the UCR path with the -p flag, e.g. -p:/path/to/dataset"); }
    UCRPATH = fs::path(parg_UCRPATH.value());
  }

  // Dataset name
  string dsname;
  {
    auto parg_UCRNAME = tempo::scli::get_parameter<string>(args, "-n", tempo::scli::extract_string);
    if (!parg_UCRNAME) { do_exit(1, "specify the UCR dataset name with the -n flag -n:NAME"); }
    dsname = std::move(parg_UCRNAME.value());
  }

  // --- --- --- --- --- ---
  // Read the dataset TRAIN.ts and TEST.ts files
  fs::path dspath = UCRPATH/dsname;

  DTS train_split = std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TRAIN.ts"), "train"));
  DatasetHeader const& train_header = train_split.header();
  size_t train_top = train_header.size();

  DTS test_split =
    std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TEST.ts"), "test", train_header.label_encoder()));
  DatasetHeader const& test_header = test_split.header();
  size_t test_top = test_header.size();

  // --- --- --- --- --- ---
  // Json Record
  Json::Value jv;
  jv["dataset"] = train_header.name();

  // --- --- --- --- --- ---
  // Analyse train
  {
    // --- Header info
    Json::Value v = train_header.to_json();
    v.removeMember("name");
    // --- Stats
    {
      Json::Value statsv;
      DTS_Stats stats(train_split);
      statsv["min"] = utils::to_json(stats._min);
      statsv["max"] = utils::to_json(stats._max);
      statsv["mean"] = utils::to_json(stats._mean);
      statsv["stddev"] = utils::to_json(stats._stddev);
      v["stats"] = std::move(statsv);
    }
    // --- Sample
    if (mm_nbsample>0) {
      Json::Value mmv;
      if (train_header.has_missing_value()) {
        mmv = Json::Value("error_missing_value");
      } else if(train_header.nb_dimensions() != 1) {
        mmv = Json::Value("error_multivariate");
      } else {
        mmv["size"] = mm_nbsample;
        utils::StddevWelford welford;
        size_t j = 0;
        std::uniform_int_distribution<> distrib(0, (int)train_split.size() - 1);
        for (size_t i = 0; i<mm_nbsample; ++i) {
          const auto& q = train_split[distrib(prng)];
          const auto& s = train_split[distrib(prng)];
          const auto& cost = tempo::distance::minkowski(q, s, mm_exponent);
          welford.update(cost);
        }
        mmv["mean"] = welford.get_mean();
        mmv["stddev"] = welford.get_stddev_s();
        mmv["variance"] = welford.get_variance_s();
      }
      v["modminkowki_sample"] = std::move(mmv);
    }
    // --- Return
    jv["train"] = std::move(v);

  }
  {
    Json::Value v = test_header.to_json();
    v.removeMember("name");
    jv["test"] = std::move(v);
  }

  // --- --- --- --- --- ---
  // Modified Minkowski Sampling


  // --- --- --- --- --- ---
  // Stats on train and test

  cout << endl << jv.toStyledString() << endl;
  if (outpath) {
    auto out = ofstream(outpath.value());
    out << jv << endl;
  }

  return 0;
}
