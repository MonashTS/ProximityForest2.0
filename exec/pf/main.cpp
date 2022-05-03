#include "pch.hpp"

#include <libtempo/classifier/splitting_forest/proximity_forest/pf2018.hpp>
#include <libtempo/transform/derivative.hpp>
#include <libtempo/reader/reader.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) {
    std::cerr << msg.value() << std::endl;
  }
  exit(code);
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace libtempo;
  using json = nlohmann::json;
  using DS = DTS<double>;

  std::random_device rd;

  // ARGS
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);

  // Config
  fs::path UCRPATH(args[0]);
  string dataset_name(args[1]);
  size_t nbt = 100;
  size_t nbc = 5;
  size_t nbthread = 8;
  size_t seed = rd();

  // Json record
  json j;

  // Load UCR train and test dataset
  DS train_dataset;
  DS test_dataset;
  {
    auto start = utils::now();
    { // Read train
      fs::path train_path = UCRPATH/dataset_name/(dataset_name + "_TRAIN.ts");
      auto variant_train = reader::load_dataset_ts(train_path);
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + train_path.string() + "': " + std::get<0>(variant_train)}); }
    }
    { // Read test
      fs::path test_path = UCRPATH/dataset_name/(dataset_name + "_TEST.ts");
      auto variant_test = reader::load_dataset_ts(test_path);
      if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
      else { do_exit(1, {"Could not read train set '" + test_path.string() + "': " + std::get<0>(variant_test)}); }
    }
    auto delta = utils::now() - start;
    json dataset;
    dataset["train"] = train_dataset.header().to_json();
    dataset["test"] = test_dataset.header().to_json();
    dataset["load_time_ns"] = delta.count();
    dataset["load_time_str"] = utils::as_string(delta);
    j["dataset"] = dataset;
  }


  // --- --- --- PF2018
  // --- --- train
  classifier::pf::PF2018 pf2018(nbt, nbc);
  // Make Transformation
  auto transformations = std::make_shared<classifier::pf::DatasetMap_t<double>>();
  {
    transformations->insert({"default", train_dataset});
    auto train_derives = transform::derive(train_dataset, 1);
    transformations->insert({"d1", std::move(train_derives[0])});
  }
  // actual train call.
  const auto train_start = utils::now();
  auto trained_forest = pf2018.train(seed, transformations, nbthread);
  const auto train_delta = utils::now()-train_start;

  // --- --- test
  // Make transformation
  const size_t test_seed = seed*nbc+nbt;
  auto test_transformations = std::make_shared<classifier::pf::DatasetMap_t<double>>();
  {
    test_transformations->insert({"default", test_dataset});
    auto test_derives = transform::derive(test_dataset, 1);
    test_transformations->insert({"d1", std::move(test_derives[0])});
  }
  // Get the classifier
  auto classifier = trained_forest.get_classifier_for(test_seed, test_transformations, false);
  const size_t test_top = test_dataset.header().size();
  size_t correct = 0;
  for (size_t i = 0; i<test_top; ++i) {
    arma::Col<double> proba = classifier.predict_proba(i, nbthread);
    size_t predicted_idx = std::distance(proba.begin(), std::max_element(proba.begin(), proba.end()));
    std::string predicted_l = train_dataset.header().index_to_label().at(predicted_idx);
    std::string true_l = test_dataset.header().labels()[i].value();
    if (predicted_l==true_l) { correct++; }
  }

  std::cout << "Result with " << nbt << " trees:" << std::endl;
  std::cout << "  correct: " << correct << "/" << test_top << std::endl;
  std::cout << "  accuracy: " << (double)correct/test_top << std::endl;

  // Output
  cout << j.dump(2) << endl;


  return 0;
}