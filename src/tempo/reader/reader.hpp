#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/utils/label_encoder.hpp>
#include <tempo/dataset/dts.hpp>

#include "ts/ts.hpp"

/**
 * Give access to readers
 */
namespace tempo::reader {

  /// Read a TS file - univariate only
  /// Can use an existing label encoder.
  /// TODO: fix reader for multivariate, it is broken
  std::variant<std::string, DTS> load_udataset_ts(
    std::filesystem::path const& path,
    std::string const& split_name,
    LabelEncoder const& encoder = {}
  );

  /// Read a csv file - univariate series
  /// Can use an existing label encoder.
  std::variant<std::string, DTS> load_udataset_csv(
    std::filesystem::path const& path,
    std::string const& dataset_name,
    std::string const& split_name,
    LabelEncoder const& encoder = {},
    bool csvheader = false,
    char csvsep = ',',
    std::set<char> csvcomment = {'%', '@'}
  );

} // End of namespace tempo::reader