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

  // Output file
  optional<fs::path> outpath{};
  {
    auto p_out = tempo::scli::get_parameter<string>(args, "-out", tempo::scli::extract_string);
    if (p_out) { outpath = {fs::path{p_out.value()}}; }
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
    dsname = parg_UCRNAME.value();
  }

  // --- --- --- --- --- ---
  // Read the dataset TRAIN.ts and TEST.ts files
  fs::path dspath = UCRPATH/dsname;

  DataSplit<TSeries> train_split = std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TRAIN.ts"), "train"));
  DatasetHeader const& train_header = train_split.header();
  size_t train_top = train_header.size();

  DataSplit<TSeries> test_split = std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TEST.ts"), "test", train_header.label_encoder()));
  DatasetHeader const& test_header = test_split.header();
  size_t test_top = test_header.size();

  // Json Record
  Json::Value jv;

  jv["dataset"] = train_header.name();

  // --- --- --- --- --- ---
  // Sanity check
  {
    std::vector<std::string> errors = {};

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    auto [bcm, remainder] = train_split.get_BCM();

    if (!remainder.empty()) {
      errors.emplace_back("Train set: contains exemplar without label");
    }

    for (const auto& [label, vec] : bcm) {
      if (vec.size()<2) {
        errors.emplace_back("Train set: contains a class with only one exemplar");
        break;
      }
    }
  }





  cout << endl << jv.toStyledString() << endl;
  if (outpath) {
    auto out = ofstream(outpath.value());
    out << jv << endl;
  }

  return 0;
}
