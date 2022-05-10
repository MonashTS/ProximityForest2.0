#include "pch.h"

#include <tempo/utils/utils.hpp>
#include <tempo/classifier/splitting_forest/proximity_forest/pf2018.hpp>

namespace fs = std::filesystem;


[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) {
    std::cerr << msg.value() << std::endl;
  }
  exit(code);
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;

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
  size_t train_seed = rd();
  size_t test_seed = train_seed+(nbt*nbc);

  // Json record
  Json::Value j;

  // Load UCR train and test dataset
  DTS train_dataset;
  DTS test_dataset;
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
    Json::Value dataset;
    dataset["train"] = train_dataset.header().to_json();
    dataset["test"] = test_dataset.header().to_json();
    dataset["load_time_ns"] = delta.count();
    dataset["load_time_str"] = utils::as_string(delta);
    j["dataset"] = dataset;
  }

  // --- --- --- PF2018
  // --- --- train
  classifier::PF2018 pf2018(nbt, nbc);
  {
    DTSMap trainset;
    trainset.insert({"default", train_dataset});
    auto train_derives = transform::derive(train_dataset, 1);
    trainset.insert({"d1", std::move(train_derives[0])});
    // actual train call.
    const auto train_start = utils::now();
    pf2018.train(trainset, train_seed, nbthread);
    const auto train_delta = utils::now() - train_start;
    // Record JSON
  }

  // --- --- test
  {
    // Prepare data
    DTSMap testset;
    testset.insert({"default", test_dataset});
    auto test_derives = transform::derive(test_dataset, 1);
    testset.insert({"d1", std::move(test_derives[0])});
    // Prepare output
    arma::mat probabilities;
    arma::rowvec weights;
    // Call
    const auto test_start = utils::now();
    pf2018.predict(testset, probabilities, weights, test_seed, nbthread);
    const auto test_delta = utils::now() - test_start;
    // Record JSON
    const auto& test_header = test_dataset.header();
    const size_t test_top = test_header.size();
    Json::Value result_probas;
    Json::Value result_weights;
    Json::Value result_truelabels;
    // For each query (test instance)
    for(size_t query=0; query<test_top; ++query){
      // Store the probabilities
      result_probas.append(utils::to_json(probabilities.col(query)));
      // Store the weight associated with the probabilities
      result_weights.append(Json::Value(weights[query]));
      // Store the true label
      string true_l = test_header.labels()[query].value();
      size_t true_label_idx = test_header.label_encoder().label_to_index().at(true_l);
      result_truelabels.append(true_label_idx);
    }
    j["result_probabilities"] = result_probas;
    j["result_truelabels"] = result_truelabels;
    j["result_weights"] = result_weights;
  }

  // Output
  cout << j.toStyledString() << endl;

  return 0;
}