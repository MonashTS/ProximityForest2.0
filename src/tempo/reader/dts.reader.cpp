#include "dts.reader.hpp"

namespace tempo::reader::dataset {

  Result load(std::variant<ts_ucr, csv> config) {

    // Helper function to make an error
    auto make_error = [](std::string msg) -> Result {
      return Result(std::in_place_index<0>, "Error: " + std::move(msg));
    };

    // Helper load CSV
    auto load_csv = [](csv& conf, std::filesystem::path csvpath, std::string const& split_name) {
      return load_udataset_csv(csvpath, conf.dataset_name, split_name, {}, conf.csv_skip_header,
                               conf.csv_separator);
    };

    TrainTest result;
    {
      auto start = utils::now();
      if (config.index()==0) {
        ts_ucr conf = std::get<0>(config);
        std::filesystem::path dataset_path = conf.ucr_dir/conf.name;
        // --- --- --- Read train
        {
          std::filesystem::path train_path = dataset_path/(conf.name + "_TRAIN.ts");
          auto variant_train = load_udataset_ts(train_path, "train");
          if (variant_train.index()==1) { result.train_dataset = std::get<1>(variant_train); }
          else { return make_error("train set '" + train_path.string() + "': " + std::get<0>(variant_train)); }
        }
        // --- --- --- Read test
        {
          std::filesystem::path test_path = dataset_path/(conf.name + "_TEST.ts");
          auto variant_test = load_udataset_ts(test_path, "test");
          if (variant_test.index()==1) { result.test_dataset = std::get<1>(variant_test); }
          else { return make_error("test set '" + test_path.string() + "': " + std::get<0>(variant_test)); }
        }
      } else if (config.index()==1) {
        csv conf = std::get<1>(config);
        // --- --- --- Read train
        {
          auto variant_train = load_csv(conf, conf.path_to_train, "train");
          if (variant_train.index()==1) { result.train_dataset = std::get<1>(variant_train); }
          else { return make_error("train set '" + conf.path_to_train.string() + "': " + std::get<0>(variant_train)); }
        }
        // --- --- --- Read test
        {
          auto variant_test = load_csv(conf, conf.path_to_test, "test");
          if (variant_test.index()==1) { result.test_dataset = std::get<1>(variant_test); }
          else { return make_error("test set '" + conf.path_to_test.string() + "': " + std::get<0>(variant_test)); }
        }
      } else { tempo::utils::should_not_happen(); }
      result.load_time = utils::now() - start;
    }

    return {result};
  }

  std::vector<std::string> sanity_check(TrainTest const& train_test) {
    DatasetHeader const& train_header = train_test.train_dataset.header();
    DatasetHeader const& test_header = train_test.test_dataset.header();
    auto [train_bcm, train_bcm_remains] = train_test.train_dataset.get_BCM();
    std::vector<std::string> errors = {};

    if (!train_bcm_remains.empty()) {
      errors.emplace_back("Could not take the By Class Map for all train exemplar (exemplar without label)");
    }

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    return errors;
  }

} // End of namespace tempo::reader