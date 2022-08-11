#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include "ts/ts.hpp"
#include "csv/csv.hpp"

#include <algorithm>

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

  /// Read a csv file - univariate series
  inline std::variant<std::string, DTS> load_dataset_csv(
    std::filesystem::path const& path,
    std::string dataset_name,
    size_t nb_var,
    std::string split_name,
    LabelEncoder encoder = {},
    bool csvheader = false, char csvsep = ','
  ) {

    try {
      std::ifstream input(path, std::ios::binary);
      CSVDataset csvdata = read_csv(input, csvheader, csvsep);

      // Build vector of labels and TSeries, and check for series with missing values
      std::vector<std::optional<std::string>> vlabels;
      vlabels.reserve(csvdata.rows.size());

      std::vector<TSeries> series;
      series.reserve(csvdata.rows.size());

      std::vector<size_t> series_with_missing_values;

      for (size_t i = 0; i<csvdata.rows.size(); ++i) {
        auto& r = csvdata.rows[i];
        vlabels.emplace_back(r.label);
        bool missing = std::any_of(r.data.begin(), r.data.end(), [](double d) -> bool { return std::isnan(d); });
        // Check missing data
        if (missing) { series_with_missing_values.push_back(i); }
        // Build tseries
        series.push_back(TSeries::mk_from_rowmajor(std::move(r.data), nb_var, {vlabels.back()}, {missing}));
      }

      // Build Header
      std::shared_ptr<DatasetHeader> header = std::make_shared<DatasetHeader>(
        dataset_name,
        csvdata.length_min,
        csvdata.length_max,
        nb_var,
        std::move(vlabels),
        std::move(series_with_missing_values),
        std::move(encoder)
      );

      // Build Transform (raw data, "default")
      std::shared_ptr<DatasetTransform<TSeries>> rawd = std::make_shared<DatasetTransform<TSeries>>(
        std::move(header),
        "default",
        std::move(series)
      );

      // Build and return the split
      return {DataSplit<TSeries>(std::move(split_name), std::move(rawd))};
    } catch (std::exception& e) {
      return {e.what()};
    }

  }

} // End of namespace tempo::reader