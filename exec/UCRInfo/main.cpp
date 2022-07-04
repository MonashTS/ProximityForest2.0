#include "pch.h"
#include <typeinfo>
#include <tempo/reader/new_reader.hpp>

#include "cli.hpp"

namespace fs = std::filesystem;

struct config {
  int nbsample{0};
  std::vector<double> exponents{};
};

// Given a UCR dataset in ts format, compute some information
Json::Value compute_info(
  tempo::DatasetHeader const& header,
  tempo::DTS const& dts,
  config const& conf,
  tempo::PRNG& prng
) {
  using namespace tempo;
  // --- Header info
  Json::Value v = header.to_json();
  // --- Stats
  {
    Json::Value statsv;
    DTS_Stats stats(dts);
    statsv["min"] = utils::to_json(stats._min);
    statsv["max"] = utils::to_json(stats._max);
    statsv["mean"] = utils::to_json(stats._mean);
    statsv["stddev"] = utils::to_json(stats._stddev);
    v["stats"] = std::move(statsv);
  }
  // --- Sample
  if (conf.nbsample>0) {
    Json::Value mmv;
    if (header.has_missing_value()) { mmv = Json::Value("error_missing_value"); }
    else if (header.nb_dimensions()!=1) { mmv = Json::Value("error_multivariate"); }
    else {
      mmv["results"] = Json::arrayValue;
      for (const double exponent : conf.exponents) {
        Json::Value emmv;
        emmv["sample_size"] = conf.nbsample;
        emmv["exponent"] = exponent;
        auto cf = tempo::distance::univariate::ade<tempo::TSeries>(exponent);
        utils::StddevWelford welford;
        size_t j = 0;
        std::uniform_int_distribution<> distrib(0, (int)dts.size() - 1);
        for (int i = 0; i<conf.nbsample; ++i) {
          const auto& q = dts[distrib(prng)];
          const auto& s = dts[distrib(prng)];
          const auto& cost = tempo::distance::directa(q, s, cf, tempo::utils::PINF);
          welford.update(cost);
        }
        emmv["mean"] = welford.get_mean();
        emmv["stddev"] = welford.get_stddev_s();
        mmv["results"].append(std::move(emmv));
      }
    }
    v["direct_alignment_sample"] = std::move(mmv);
  }
  // --- Check for duplicated series
  {
    std::map<size_t, std::vector<size_t>> same;
    std::set<size_t> visited;
    for (size_t i = 0; i<dts.size(); ++i) {
      TSeries const& ref = dts[i];
      for (size_t j = i + 1; j<dts.size(); ++j) {
        if(visited.contains(j)){continue;}
        TSeries const& other = dts[j];
        if (arma::norm(ref.data() - other.data())==0) {
          visited.insert(j);
          same[i].push_back(j);
        }
      }
    }
    Json::Value duplicated = Json::arrayValue;
    size_t n=0;
    for (auto [k, vec] : same) {
      vec.insert(vec.begin(), k);
      n+=vec.size();
      duplicated.append(utils::to_json(vec));
    }
    v["duplicated_nb"] = (int)n;
    v["duplicated"] = duplicated;

  }
  return v;
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;
  using namespace tempo::utils;

  // --- --- --- --- --- ---
  // Program Arguments
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if (args.empty()) { do_exit(0, usage); }
  config conf;

  // --- --- ---
  // Optional

  // Random seed for sampling
  long long seed;
  {
    auto p_seed = tempo::scli::get_parameter<long long>(args, "-seed", tempo::scli::extract_int, -1);
    if (p_seed<0) { seed = std::random_device()(); } else { seed = p_seed; }
  }
  tempo::PRNG prng(seed);

  // Output file
  optional<fs::path> outpath{};
  {
    auto p_out = tempo::scli::get_parameter<string>(args, "-out", tempo::scli::extract_string);
    if (p_out) { outpath = {fs::path{p_out.value()}}; }
  }

  // Sample with direct alignment
  {
    auto parg_dist = tempo::scli::get_parameter<string>(args, "-sample", tempo::scli::extract_string);
    if (parg_dist) {
      auto v = tempo::reader::split(parg_dist.value(), ':');
      bool ok = v.size()>=2;
      if (ok) {
        auto on = tempo::reader::as_int(v[0]);
        ok = (bool)on;
        if (ok) { conf.nbsample = on.value(); }
        //
        for (size_t i = 1; ok&&i<v.size(); ++i) {
          auto oe = tempo::reader::as_double(v[i]);
          ok = (bool)oe;
          if (ok) { conf.exponents.push_back(oe.value()); }
        }
      }
      if (!ok) { do_exit(1, "specify <n>:<e0>:...:<en> after '-sample'"); }
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

  DTS train_split = std::get<1>(tempo::reader::load_dataset_ts(dspath/(dsname + "_TRAIN.ts"), "train"));
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
  // Compute Info for train and test
  {
    Json::Value train_v = compute_info(train_header, train_split, conf, prng);
    train_v.removeMember("name");
    jv["train"] = std::move(train_v);
  }
  {
    Json::Value test_v = compute_info(test_header, test_split, conf, prng);
    test_v.removeMember("name");
    jv["test"] = std::move(test_v);
  }

  // --- --- --- --- --- ---
  // Print
  cout << endl << jv.toStyledString() << endl;
  if (outpath) {
    auto out = ofstream(outpath.value());
    out << jv << endl;
  }

  return 0;
}
