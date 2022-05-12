#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>
#include <tempo/tseries/dataset.hpp>

#include "ts/ts.hpp"


namespace tempo::reader {

  /// Helper for TS file format and path
  inline std::variant<std::string, Dataset<TSeries>>
  load_dataset_ts(const std::filesystem::path& path){
    std::ifstream istream_(path);
    return tempo::reader::TSReader::read(istream_);
  }

  /// Helper for TS file format and path
  inline std::variant<std::string, Dataset<TSeries>>
  load_dataset_ts(const std::filesystem::path& path, LabelEncoder const& encoder){
    std::ifstream istream_(path);
    return tempo::reader::TSReader::read(istream_, {{encoder}});
  }

} // End of namespace tempo::reader
