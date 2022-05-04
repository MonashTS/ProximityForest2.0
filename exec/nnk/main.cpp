#include "pch.hpp"

#include <libtempo/transform/derivative.hpp>
#include <libtempo/reader/reader.hpp>

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
  using PRNG = mt19937_64;

  std::random_device rd;

  // ARGS
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);

  // Config
  fs::path UCRPATH(args[0]);
  string dataset_name(args[1]);
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

  json results;
  const auto& test_header = test_dataset.header();
  const auto& train_header = train_dataset.header();
  IndexSet train_is = train_header.index_set();
  vector<size_t> idxs = train_is.vector();
  std::shuffle(std::begin(idxs), std::end(idxs), PRNG(rd()));
  const size_t test_top = test_header.size();
  const auto test_start = utils::now();

  struct NN {
    size_t cidx;
    double distance;
  };

  for (size_t q = 0; q<test_top; ++q) {
    // --- --- ---
    for(const auto c: idxs){
      std::cout << c << std::endl;
    }
    // --- --- ---
  }
  const auto test_delta = utils::now() - test_start;
  j["results"] = results;

  // Output
  cout << j.dump(2) << endl;

  return 0;
}