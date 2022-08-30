#include <tempo/utils/readingtools.hpp>
#include <tempo/reader/new_reader.hpp>
#include <tempo/transform/univariate.hpp>
#include <tempo/classifier/ProximityForest/pf2018.hpp>
#include "cmdline.hpp"

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) { std::cerr << msg.value() << std::endl; }
  exit(code);
}

int main(int argc, char **argv) {
  // --- --- --- Type / namespace
  using namespace std;
  using namespace tempo;

  // --- --- --- Randomness
  std::random_device rd;
  size_t train_seed = rd();
  size_t test_seed = rd();
  size_t tiebreak_seed = rd();

  // --- --- --- Prepare JSon record for output
  Json::Value jv;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Read args
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  cmdopt opt;
  {
    variant<string, cmdopt> mb_opt = parse_cmd(argc, argv);
    switch (mb_opt.index()) {
    case 0: {
      cerr << "Error: " << std::get<0>(mb_opt) << std::endl;
      exit(1);
    }
    case 1: { opt = std::get<1>(mb_opt); }
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Check PF configuration
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // TODO
  // To link with transforms, splitter requirement, etc...

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Read dataset
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- Load UCR train and test dataset
  DTS train_dataset;
  DTS test_dataset;
  {
    auto start = utils::now();

    if (opt.input.index()==0) {
      read_ucr ru = std::get<0>(opt.input);
      // --- --- --- Read train
      auto variant_train = tempo::reader::load_dataset_ts(ru.train, "train");
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + ru.train.string() + "': " + std::get<0>(variant_train)}); }
      // --- --- --- Read test
      auto variant_test = tempo::reader::load_dataset_ts(ru.test, "test");
      if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
      else { do_exit(1, {"Could not read train set '" + ru.test.string() + "': " + std::get<0>(variant_test)}); }
    } else if (opt.input.index()==1) {
      read_csv rc = std::get<1>(opt.input);
      // --- --- --- Read train
      auto variant_train = tempo::reader::load_dataset_csv(rc.train, rc.name, 1, "train", {}, rc.skip_header, rc.sep);
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + rc.train.string() + "': " + std::get<0>(variant_train)}); }
      // --- --- --- Read test
      auto variant_test = tempo::reader::load_dataset_csv(rc.test, rc.name, 1, "test", {}, rc.skip_header, rc.sep);
      if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
      else { do_exit(1, {"Could not read test set '" + rc.test.string() + "': " + std::get<0>(variant_test)}); }
    } else { tempo::utils::should_not_happen(); }

    auto delta = utils::now() - start;
    Json::Value dataset;
    dataset["train"] = train_dataset.header().to_json();
    dataset["test"] = test_dataset.header().to_json();
    dataset["load_time_ns"] = delta.count();
    dataset["load_time_str"] = utils::as_string(delta);
    jv["dataset"] = dataset;
  } // End of dataset loading

  DatasetHeader const& train_header = train_dataset.header();
  DatasetHeader const& test_header = test_dataset.header();

  // --- --- --- Sanity check
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

    if (!errors.empty()) {
      jv["status"] = "error";
      jv["status_message"] = utils::cat(errors, "; ");
      cout << jv.toStyledString() << endl;
      if (opt.output) {
        auto out = ofstream(opt.output.value());
        out << jv << endl;
      }
      exit(1);
    }

  } // End of Sanity Check


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Prepare data according to the splitters requirement
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  map<string, DTS> train_map;
  map<string, DTS> test_map;

  {
    namespace ttu = tempo::transform::univariate;

    // --- TRAIN
    auto train_derive_t1 = train_dataset.transform().map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), "d1");
    auto train_derive_t2 = train_derive_t1->map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), "d2");
    DTS train_derive_1("train", train_derive_t1);
    DTS train_derive_2("train", train_derive_t2);
    train_map["default"] = train_dataset;
    train_map["derivative1"] = train_derive_1;
    train_map["derivative2"] = train_derive_2;

    // --- TEST
    auto test_derive_t1 = test_dataset.transform().map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), "d1");
    auto test_derive_t2 = test_derive_t1->map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), "d2");
    DTS test_derive_1("test", test_derive_t1);
    DTS test_derive_2("test", test_derive_t2);
    test_map["default"] = test_dataset;
    test_map["derivative1"] = test_derive_1;
    test_map["derivative2"] = test_derive_2;
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Call PF
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  tempo::classifier::PF2018 pf(opt.nb_trees, opt.nb_candidates, opt.pfconfig);

  // Train
  auto train_start_time = utils::now();
  pf.train(train_map, train_seed, opt.nb_threads);
  auto train_elapsed = utils::now() - train_start_time;

  // Test
  auto test_start_time = utils::now();
  classifier::ResultN result = pf.predict(test_map, test_seed, opt.nb_threads);
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
    jv["classifier"] = opt.pfconfig;
    jv["classifier_info"] = j;
  }

  { // 01 loss results
    Json::Value j;
    j["nb_corrects"] = nb_correct;
    j["accuracy"] = accuracy;
    jv["01loss"] = j;
  }

  cout << jv << endl;

  if (opt.output) {
    auto out = ofstream(opt.output.value());
    out << jv << endl;
  }

  return 0;

}