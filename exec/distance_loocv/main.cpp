#include <string>
#include <vector>

#include <tempo/utils/utils.hpp>
#include <tempo/reader/new_reader.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/loocv/partable/partable.hpp>

#include "cli.hpp"

int main(int argc, char **argv) {
  using namespace std;

  const size_t SAMPLE_SIZE = 4000;
  const double omega_exp = 5;

  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if (argc<6) {
    cout << "<path to ucr> <dataset name> <ge> <nbthreads> <output> required" << endl;
    exit(1);
  }

  // --- --- ---
  // Check arguments

  random_device rd;
  tempo::PRNG prng(rd());

  vector<string> argList(argv, argv + argc);
  filesystem::path path_ucr(argList[1]);
  string dataset_name(argList[2]);
  const double ge = stod(argList[3]);
  size_t nbthreads = stoi(argList[4]);
  filesystem::path outpath(argList[5]);

  // --- --- ---
  // Prepare result
  Json::Value jv;
  std::ofstream outfile(outpath);

  // --- --- ---
  // Load the datasets
  tempo::DTS loaded_train_split;
  tempo::DTS loaded_test_split;
  {
    filesystem::path path_train = path_ucr/dataset_name/(dataset_name + "_TRAIN.ts");
    filesystem::path path_test = path_ucr/dataset_name/(dataset_name + "_TEST.ts");
    auto train = tempo::reader::load_dataset_ts(path_train, "train");
    if (train.index()==0) { do_exit(2, "Could not load the train set: " + std::get<0>(train)); }
    loaded_train_split = std::move(std::get<1>(train));
    // Load test set
    auto test = tempo::reader::load_dataset_ts(path_test, "test", loaded_train_split.header().label_encoder());
    if (test.index()==0) { do_exit(2, "Could not load the test set: " + std::get<0>(test)); }
    loaded_test_split = std::move(std::get<1>(test));
  }
  tempo::DatasetHeader const& train_header = loaded_train_split.header();
  tempo::DatasetHeader const& test_header = loaded_test_split.header();

  jv["train"] = train_header.to_json();
  jv["test"] = test_header.to_json();

  // --- --- ---
  // Sanity check for the dataset
  {
    std::vector<std::string> errors = {};

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    auto [bcm, remainder] = loaded_train_split.get_BCM();

    if (!remainder.empty()) {
      errors.emplace_back("Train set: contains exemplar without label");
    }

    for (const auto& [label, vec] : bcm) {
      if (vec.size()<2) {
        errors.emplace_back("Train set: contains a class with only one exemplar");
        break;
      }
    }

    if (!errors.empty()) {
      jv["status"] = "error";
      jv["status_message"] = tempo::utils::cat(errors, "; ");
      cout << jv.toStyledString() << endl;
      outfile << jv << endl;
      exit(2);
    }
  }




  // --- --- ---
  // Sample ADTW




  return 0;
}
