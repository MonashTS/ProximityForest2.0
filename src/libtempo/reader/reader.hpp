#pragma once

#include <libtempo/tseries/tseries.hpp>
#include <libtempo/tseries/dataset.hpp>

#include "ts/ts.hpp"

#include <filesystem>

namespace libtempo::reader {

  /// Helper for TS file format and path
  inline std::variant<std::string, Dataset<TSeries<double>>> load_dataset_ts(const std::filesystem::path& path){
    std::ifstream istream_(path);
    return libtempo::reader::TSReader::read(istream_);
  }

} // End of namespace libtempo::reader
