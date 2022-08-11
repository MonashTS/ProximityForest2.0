#include "pch.h"

#include <cxxopts.hpp>
#include <tempo/utils/readingtools.hpp>

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
  cxxopts::Options options("Proximity Forest", "Proximity Forest Time Series Classifier");
  options.add_options()
    ("ucr", "<path to UCR>", cxxopts::value<std::string>())
    ("train", "<path to csv>", cxxopts::value<std::string>())
    ("test", "<path to csv>", cxxopts::value<std::string>())
    ("dataset_name", "Name of the dataset", cxxopts::value<std::string>())
    ("pfversion", "which PF to use", cxxopts::value<std::string>())
    ("nb_trees", "number of trees", cxxopts::value<size_t>())
    ("nb_candidates", "number of candidates", cxxopts::value<size_t>())
    ("nb_threads", "number of threads", cxxopts::value<size_t>())
    ("output", "output file", cxxopts::value<std::string>());

  options.parse_positional({"dataset_name", "pfversion", "nb_trees", "nb_candidates", "nb_threads", "output"});
  options.positional_help("<dataset_name> <pfversion> <nb_trees> <nb_candidates> <nb_threads> [output]");

  // Get results
  enum IType { NONE, UCR, CSV };
  IType itype{NONE};
  fs::path UCRPATH;
  fs::path csvtrain;
  fs::path csvtest;
  string dataset_name;
  string pfversion;
  size_t nb_trees;
  size_t nb_candidates;
  size_t nb_threads;
  optional <fs::path> outpath{};
  bool csv_header=false;
  char csv_sep=' ';


  try {
    auto cliopt = options.parse(argc, argv);

    // Get UCR
    if (cliopt.count("ucr")) {
      UCRPATH = cliopt["ucr"].as<std::string>();
      itype = UCR;
    }

    // GET CSV
    bool hastrain = cliopt.count("train")>0;
    bool hastest = cliopt.count("test")>0;
    if (hastrain ^ hastest) {
      throw std::runtime_error("'--train' and '--test' must both be specified together");
    }
    if (hastrain & hastest) {
      if (itype==UCR) {
        throw std::runtime_error("'--ucr' must appear without '--train' and '--test'");
      }
      csvtrain = fs::path(cliopt["train"].as<std::string>());
      csvtest = fs::path(cliopt["test"].as<std::string>());
      itype = CSV;
    }

    // Other positional arguments
    dataset_name = cliopt["dataset_name"].as<std::string>();
    pfversion = cliopt["pfversion"].as<std::string>();
    nb_trees = cliopt["nb_trees"].as<size_t>();
    nb_candidates = cliopt["nb_candidates"].as<size_t>();
    nb_threads = cliopt["nb_threads"].as<size_t>();

    // Optional output file
    if (cliopt.count("output")) { outpath = fs::path(cliopt["output"].as<std::string>()); }

  } catch (std::exception const& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << options.help() << std::endl;
    exit(1);
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

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    if (itype==UCR) {
      { // --- --- --- Read train
        fs::path train_path = UCRPATH/dataset_name/(dataset_name + "_TRAIN.ts");
        auto variant_train = tempo::reader::load_dataset_ts(train_path, "train");
        if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
        else { do_exit(1, {"Could not read train set '" + train_path.string() + "': " + std::get<0>(variant_train)}); }
      }
      { // --- --- --- Read test
        fs::path test_path = UCRPATH/dataset_name/(dataset_name + "_TEST.ts");
        auto variant_test = tempo::reader::load_dataset_ts(test_path, "test");
        if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
        else { do_exit(1, {"Could not read train set '" + test_path.string() + "': " + std::get<0>(variant_test)}); }
      }
    }
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    else if (itype==CSV) {
      { // --- --- --- Read train
        auto variant = tempo::reader::load_dataset_csv(csvtrain, dataset_name, 1, "train", {}, csv_header, csv_sep);
        if (variant.index()==1) { train_dataset = std::get<1>(variant); }
        else { do_exit(1, {"Could not read train set '" + csvtrain.string() + "': " + std::get<0>(variant)}); }
      }
      { // --- --- --- Read test
        auto variant = tempo::reader::load_dataset_csv(csvtest, dataset_name, 1, "test", {}, csv_header, csv_sep);
        if (variant.index()==1) { test_dataset = std::get<1>(variant); }
        else { do_exit(1, {"Could not read test set '" + csvtest.string() + "': " + std::get<0>(variant)}); }
      }
    }
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    else { tempo::utils::should_not_happen(); }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

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
  map <string, DTS> train_map;
  map <string, DTS> test_map;
  {
    auto train_derive_all = tempo::transform::derive(train_dataset.transform(), 2);
    auto train_derive_t1 = std::make_shared<DatasetTransform<TSeries>>
    (std::move(train_derive_all[0]));
    auto train_derive_t2 = std::make_shared<DatasetTransform<TSeries>>
    (std::move(train_derive_all[1]));

    DTS train_derive_1("train", train_derive_t1);
    DTS train_derive_2("train", train_derive_t2);

    // Train map
    train_map["default"] = train_dataset;
    train_map["derivative1"] = train_derive_1;
    train_map["derivative2"] = train_derive_2;

    // TEST

    auto test_derive_all = tempo::transform::derive(test_dataset.transform(), 2);
    auto test_derive_t1 = std::make_shared<DatasetTransform<TSeries>>
    (std::move(test_derive_all[0]));
    auto test_derive_t2 = std::make_shared<DatasetTransform<TSeries>>
    (std::move(test_derive_all[1]));

    DTS test_derive_1("test", test_derive_t1);
    DTS test_derive_2("test", test_derive_t2);

    // Test map
    test_map["default"] = test_dataset;
    test_map["derivative1"] = test_derive_1;
    test_map["derivative2"] = test_derive_2;

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