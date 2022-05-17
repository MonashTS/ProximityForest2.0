#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/tseries/tseries.hpp>
#include <tempo/tseries/dataset.hpp>

#include "ts/ts.hpp"

namespace tempo::reader {

  /// Helper for TS file format and path
  inline std::variant<std::string, TSData>
  load_tsdata(const std::filesystem::path& path) {
    std::ifstream istream_(path);
    return tempo::reader::TSReader::read(istream_);
  }

  /// Helper for TS file format and path, with an optional existing encoder
  inline std::variant<std::string, Dataset<TSeries>>
  load_dataset_ts(const std::filesystem::path& path,
                  std::optional<std::reference_wrapper<LabelEncoder const>> mbencoder = {}) {
    auto vts = load_tsdata(path);
    if (vts.index()==1) {
      TSData tsdata = std::move(std::get<1>(vts));
      auto [hd, v] = tsdata.to_datasetheader(mbencoder);
      auto header = std::make_shared<DatasetHeader>(std::move(hd));
      return {DTS(move(header), "default", std::move(v))};
    } else {
      return {std::get<0>(vts)};
    }
  }


  /// Helper for the UCR archive
  struct UCRDataset {
    DTS alldata;
    size_t default_train_size;
    size_t default_test_size;
    tempo::utils::duration_t load_time_ns;

    IndexSet trainIS() const { return IndexSet(default_train_size); }

    IndexSet testIS() const { return IndexSet(default_train_size, default_test_size); }

    Json::Value to_json();
  };

  std::variant<std::string, UCRDataset> load_ucr_ts(std::filesystem::path const& ucrpath,
                                                    std::string const& datasetname);

} // End of namespace tempo::reader
