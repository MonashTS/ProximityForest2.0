#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include "ts/ts.hpp"

namespace tempo::reader {

  /// Helper for TS file format and path, with an existing encoder (empty by default)
  inline std::variant<std::string, DTS>
  load_dataset_ts(const std::filesystem::path& path, std::string split_name, LabelEncoder encoder = {}) {
    std::variant<std::string, TSData> vts = load_tsdata(path);

    if (vts.index()==1) {
      TSData tsdata = std::move(std::get<1>(vts));

      // Build label vector
      std::vector<std::optional<std::string>> vlabels;
      vlabels.reserve(tsdata.series.size());
      for (const auto& ts : tsdata.series) { vlabels.emplace_back(ts.label()); }

      // Build Header
      std::shared_ptr<DatasetHeader> header = std::make_shared<DatasetHeader>(
        tsdata.problem_name.value_or("Anonymous"),
        tsdata.shortest_length,
        tsdata.longest_length,
        tsdata.nb_dimensions,
        std::move(vlabels),
        std::move(tsdata.series_with_missing_values),
        std::move(encoder)
      );

      // Build Transform (raw data, "default")
      std::shared_ptr<DatasetTransform<TSeries>> rawd = std::make_shared<DatasetTransform<TSeries>>(
        std::move(header),
        "default",
        std::move(tsdata.series)
      );

      // Build and return the split
      return {DataSplit<TSeries>(std::move(split_name), std::move(rawd))};

    } else {
      return {std::get<0>(vts)};
    }
  }

} // End of namespace tempo::reader