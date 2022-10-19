#pragma once

#include "reader.hpp"
#include <tempo/dataset/dts.hpp>

/**
 * High level dataset reader expecting both a train and test split
 */
namespace tempo::reader::dataset {

  /// Configuration to read a train/test dataset from two CSV files
  struct csv {
    std::filesystem::path path_to_train{};
    std::filesystem::path path_to_test{};
    std::string dataset_name;
    bool csv_skip_header;
    char csv_separator;
  };

  /// Configuration to read train/test dataset from a TS UCR archive
  struct ts_ucr {
    std::filesystem::path ucr_dir;
    std::string name;
  };

  /// Result when successfully reading a dataset
  struct TrainTest {
    DTS train_dataset;
    DTS test_dataset;
    tempo::utils::duration_t load_time;
  };

  /// Result of loading a train/test dataset. On failure, return a string, else, return a TrainTest
  using Result = std::variant<std::string, TrainTest>;

  /// Load a train/test dataset from a configuration
  Result load(std::variant<ts_ucr, csv> config);

  /// Basic check on the dataset
  /// Return a vector of messages, each one being an error:
  /// * "Could not take the By Class Map for all train exemplar (exemplar without label)"
  /// * "Train set: variable length or missing data"
  /// * "Test set: variable length or missing data"
  std::vector<std::string> sanity_check(TrainTest const& train_test);

} // End of namespace tempo::reader