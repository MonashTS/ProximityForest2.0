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

  // Config
  fs::path UCRPATH(args[0]);
  string dataset_name(args[1]);
  string pfversion(args[2]);
  size_t nb_threads = std::stoi(args[3]);
  size_t nbt = 100;
  size_t nbc = 5;
  size_t train_seed = rd();
  size_t test_seed = train_seed + (nbt*nbc);

  // Json record
  Json::Value j;

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
    j["dataset"] = dataset;
  }

  // Get headers
  DatasetHeader const& train_header = train_dataset.header();
  DatasetHeader const& test_header = test_dataset.header();

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

  tempo::classifier::PF2018 pf(nbt, nbc, pfversion);

  // Train
  pf.train(train_map, train_seed, nb_threads);

  // Test
  classifier::ResultN result = pf.predict(test_map, test_seed, nb_threads);

  PRNG prng(rd());
  size_t nb_correct = result.nb_correct_01loss(test_header, IndexSet(test_header.size()), prng);

  cout << "Nb correct = " << nb_correct << endl;
  cout << "Accuracy   = " << (double)nb_correct/(double)test_header.size() << endl;

  return 0;

}