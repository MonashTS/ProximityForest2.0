#include "reader.hpp"
#include <filesystem>

Json::Value tempo::reader::Loaded_UCRDataset::to_json() {
  Json::Value j;
  j["train"] = train_dataset.header().to_json();
  j["test"] = test_dataset.header().to_json();
  j["load_time_ns"] = load_time_ns.count();
  j["load_time_str"] = utils::as_string(load_time_ns);
  return j;
}

std::variant<std::string, tempo::reader::Loaded_UCRDataset>
tempo::reader::load_ucr_ts(std::filesystem::path const& ucrpath, std::string const& datasetname) {
  namespace fs = std::filesystem;

  DTS train_dataset;
  DTS test_dataset;

  fs::path train_path = ucrpath/datasetname/(datasetname + "_TRAIN.ts");
  fs::path test_path = ucrpath/datasetname/(datasetname + "_TEST.ts");

  auto start = utils::now();
  // Read train
  auto variant_train = load_dataset_ts(train_path);
  if (variant_train.index()==1) { train_dataset = std::move(std::get<1>(variant_train)); }
  else { return {"Could not read train set '" + train_path.string() + "': " + std::get<0>(variant_train)}; }

  // Read test using the label encoder from train
  auto variant_test = reader::load_dataset_ts(test_path, train_dataset.header().label_encoder());
  if (variant_test.index()==1) { test_dataset = std::move(std::get<1>(variant_test)); }
  else { return {"Could not read test set '" + test_path.string() + "': " + std::get<0>(variant_test)}; }

  auto delta = utils::now() - start;

  return {Loaded_UCRDataset{
    std::move(train_dataset),
    std::move(test_dataset),
    delta
  }};

}
