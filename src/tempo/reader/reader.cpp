#include "reader.hpp"
#include <filesystem>

Json::Value tempo::reader::UCRDataset::to_json() {
  Json::Value j;
  return j;
  /*
  j["train"] = train_dataset.header().to_json();
  j["test"] = test_dataset.header().to_json();
  j["load_time_ns"] = load_time_ns.count();
  j["load_time_str"] = utils::as_string(load_time_ns);
  return j;
   */
}

std::variant<std::string, tempo::reader::UCRDataset>
tempo::reader::load_ucr_ts(std::filesystem::path const& ucrpath, std::string const& datasetname) {
  namespace fs = std::filesystem;
  using namespace std;

  fs::path train_path = ucrpath/datasetname/(datasetname + "_TRAIN.ts");
  DatasetHeader train_header;
  vector<TSeries> train_data;

  fs::path test_path = ucrpath/datasetname/(datasetname + "_TEST.ts");
  DatasetHeader test_header;
  vector<TSeries> test_data;

  auto start = utils::now();

  // Read train
  auto variant_train = load_tsdata(train_path);
  if (variant_train.index()==1) {
    TSData ts = move(get<1>(variant_train));
    auto [h, d] = ts.to_datasetheader();
    train_header = move(h);
    train_data = move(d);
  } else { return {"Could not read train set '" + train_path.string() + "': " + std::get<0>(variant_train)}; }

  // Read test
  auto variant_test = load_tsdata(test_path);
  if (variant_test.index()==1) {
    TSData ts = move(get<1>(variant_test));
    auto [h, d] = ts.to_datasetheader();
    test_header = move(h);
    test_data = move(d);
  } else { return {"Could not read test set '" + test_path.string() + "': " + std::get<0>(variant_test)}; }

  auto delta = utils::now() - start;

  // Create the header
  size_t default_train_size = train_header.size();
  size_t default_test_size = test_header.size();
  string name = train_header.name();
  auto header = make_shared<DatasetHeader>(move(train_header), move(test_header), move(name));

  // Create the dataset
  move(test_data.begin(), test_data.end(), back_inserter(train_data));
  DTS alldata(move(header), "default", move(train_data));

  return {UCRDataset{
    move(alldata),
    default_train_size,
    default_test_size,
    delta
  }};

}
