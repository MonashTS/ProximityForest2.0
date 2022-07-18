#include "pch.h"
#include <tempo/reader/new_reader.hpp>

#include <tempo/transform/derivative.hpp>
#include <tempo/classifier/ProximityForest/pf2018.hpp>

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) { std::cerr << msg.value() << std::endl; }
  exit(code);
}

int main(int argc, char **argv) {

  using namespace std;
  using namespace tempo;

  std::random_device rd;

  // ARGS
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);

  if (args.size()<6) {
    cout
      << "Usage: <path to ucr directory> <dataset name> <pfversion> <number of tree> <number of candidates> <number of threads> [output]"
      << endl;
    exit(0);
  }

  // Config
  fs::path UCRPATH(args[0]);
  string dataset_name(args[1]);
  string pfversion(args[2]);
  size_t nb_trees = std::stoi(args[3]);
  size_t nb_candidates = std::stoi(args[4]);
  size_t nb_threads = std::stoi(args[5]);

  // Output path
  optional<fs::path> outpath{};
  if (args.size()>=7) {
    outpath = fs::path(args[6]);
  }

  // Random number seeds
  size_t train_seed = rd();
  size_t test_seed = rd();
  size_t tiebreak_seed = rd();

  // Prepare JSon record for output
  Json::Value jv;

  // Load UCR train and test dataset
  DTS train_dataset;
  DTS test_dataset;
  {
    auto start = utils::now();
    { // Read train
      fs::path train_path = UCRPATH/dataset_name/(dataset_name + "_TRAIN.ts");
      auto variant_train = tempo::reader::load_dataset_ts(train_path, "train");
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + train_path.string() + "': " + std::get<0>(variant_train)}); }
    }
    { // Read test
      fs::path test_path = UCRPATH/dataset_name/(dataset_name + "_TEST.ts");
      auto variant_test = tempo::reader::load_dataset_ts(test_path, "test");
      if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
      else { do_exit(1, {"Could not read train set '" + test_path.string() + "': " + std::get<0>(variant_test)}); }
    }
    auto delta = utils::now() - start;
    Json::Value dataset;
    dataset["train"] = train_dataset.header().to_json();
    dataset["test"] = test_dataset.header().to_json();
    dataset["load_time_ns"] = delta.count();
    dataset["load_time_str"] = utils::as_string(delta);
    jv["dataset"] = dataset;
  }

  // Get headers
  DatasetHeader const& train_header = train_dataset.header();
  DatasetHeader const& test_header = test_dataset.header();

  // Sanity check
  {
    std::vector<std::string> errors = {};

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    auto [bcm, remainder] = train_dataset.get_BCM();

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
      jv["status_message"] = utils::cat(errors, "; ");
      cout << jv.toStyledString() << endl;
      if (outpath) {
        auto out = ofstream(outpath.value());
        out << jv << endl;
      }
      exit(1);
    }

  } // End of Sanity Check

  // Compute maps
  map<string, DTS> train_map;
  map<string, DTS> test_map;
  {
    // Train transform derivative1, into DTS "train"
    auto train_derive_t = std::make_shared<DatasetTransform<TSeries>>(
      std::move(tempo::transform::derive(train_dataset.transform(), 1).back())
    );
    DTS train_derive("train", train_derive_t);

    // Train map
    train_map["default"] = train_dataset;
    train_map["derivative1"] = train_derive;

    // Test transform derivative1, into DTS "test"
    auto test_derive_t = std::make_shared<DatasetTransform<TSeries>>(
      std::move(tempo::transform::derive(test_dataset.transform(), 1).back())
    );
    DTS test_derive("test", test_derive_t);

    // Test map
    test_map["default"] = test_dataset;
    test_map["derivative1"] = test_derive;
  }

  tempo::classifier::PF2018 pf(nb_trees, nb_candidates, pfversion);

  // Train
  auto train_start_time = utils::now();
  pf.train(train_map, train_seed, nb_threads);
  auto train_elapsed = utils::now() - train_start_time;

  // Test
  auto test_start_time = utils::now();
  classifier::ResultN result = pf.predict(test_map, test_seed, nb_threads);
  auto test_elapsed = utils::now() - test_start_time;

  PRNG prng(tiebreak_seed);
  size_t nb_correct = result.nb_correct_01loss(test_header, IndexSet(test_header.size()), prng);
  double accuracy = (double)nb_correct/(double)test_header.size();

  jv["status"] = "success";

  { // Classifier information
    Json::Value j;
    j["train_time_ns"] = train_elapsed.count();
    j["train_time_human"] = utils::as_string(train_elapsed);
    j["test_time_ns"] = test_elapsed.count();
    j["test_time_human"] = utils::as_string(test_elapsed);
    //
    jv["classifier"] = pfversion;
    jv["classifier_info"] = j;
  }

  { // 01 loss results
    Json::Value j;
    j["nb_corrects"] = nb_correct;
    j["accuracy"] = accuracy;
    jv["01loss"] = j;
  }

  cout << jv << endl;

  if (outpath) {
    auto out = ofstream(outpath.value());
    out << jv << endl;
  }

  return 0;

}