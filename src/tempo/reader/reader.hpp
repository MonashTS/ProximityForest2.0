#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>
#include <tempo/tseries/dataset.hpp>

#include "ts/ts.hpp"

namespace tempo::reader {

  /// Helper for TS file format and path
  inline std::variant<std::string, Dataset<TSeries>>
  load_dataset_ts(const std::filesystem::path& path) {
    std::ifstream istream_(path);
    return tempo::reader::TSReader::read(istream_);
  }

  /// Helper for TS file format and path
  inline std::variant<std::string, Dataset<TSeries>>
  load_dataset_ts(const std::filesystem::path& path, LabelEncoder const& encoder) {
    std::ifstream istream_(path);
    return tempo::reader::TSReader::read(istream_, {{encoder}});
  }

  /// Helper for the UCR archive
  struct Loaded_UCRDataset {
    DTS train_dataset;
    DTS test_dataset;
    tempo::utils::duration_t load_time_ns;

    Json::Value to_json();
  };

  std::variant<std::string, Loaded_UCRDataset> load_ucr_ts(std::filesystem::path const& ucrpath,
                                                           std::string const& datasetname);

} // End of namespace tempo::reader
