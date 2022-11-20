#include "reader.hpp"

#include "reader_result.hpp"
#include "csv/univariate/csv.hpp"
#include "ts/ts.hpp"


namespace tempo::reader {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Read a TS file - univariate only
  /// Can use an existing label encoder.
  /// TODO: fix reader for multivariate, it is broken
  std::variant<std::string, DTS> load_udataset_ts(
    std::filesystem::path const& path,
    std::string const& split_name,
    LabelEncoder const& encoder
  ) {
    if( !(std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) ){
      return {"Could not read file " + path.string()};
    }

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


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::variant<std::string, DTS> load_udataset_csv(
    std::filesystem::path const& path,
    std::string const& dataset_name,
    std::string const& split_name,
    LabelEncoder const& encoder,
    bool csvheader,
    char csvsep,
    std::set<char> csvcomment
  ) {

    if( !(std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) ){
      return {"Could not read file " + path.string()};
    }

    using namespace univariate;
    constexpr size_t nb_var = 1; // Univariate

    // Make parameters for the CSV reader
    CSVReaderParam params(CSVLabel::FIRST, csvheader, csvsep, std::move(csvcomment));

    try {

      // Open the file and read
      std::ifstream input(path, std::ios::binary);
      Result<F> csv_result = read_csv<F>(input, encoder, params);

      // Check error
      if (csv_result.index()==0) { return {std::get<0>(csv_result)}; }

      tempo::reader::ReaderData<F> csvdata = std::get<1>(std::move(csv_result));

      // Build vector of optional labels for the DatasetHeader
      std::vector<std::optional<std::string>> vlabels;
      vlabels.reserve(csvdata.series.size());

      // Build vector of index of series with missing data for the DatasetHeader
      std::vector<size_t> series_with_missing_values;

      // Build vector of TSeries for the dataset transform & dataset
      std::vector<TSeries> series;
      series.reserve(csvdata.series.size());

      for (size_t i = 0; i<csvdata.series.size(); ++i) {
        // --- --- --- Label
        if (csvdata.labels) {
          vlabels.emplace_back(csvdata.labels.value()[i]);
        } else {
          vlabels.emplace_back(std::nullopt);
        }
        // --- --- --- Missing
        bool missing = csvdata.series_with_nan.contains(i);
        if (missing) { series_with_missing_values.push_back(i); }
        // --- --- --- Build the series
        std::vector<F> data = std::move(csvdata.series[i]);
        series.push_back(TSeries::mk_from_rowmajor(std::move(data), nb_var, vlabels.back(), {missing}));
      }

      // Build Header
      std::shared_ptr<DatasetHeader> header = std::make_shared<DatasetHeader>(
        dataset_name,
        csvdata.length_min,
        csvdata.length_max,
        nb_var,
        std::move(vlabels),
        std::move(series_with_missing_values),
        std::move(csvdata.encoder)
      );

      // Build Transform (raw data, "default")
      std::shared_ptr<DatasetTransform<TSeries>> rawd = std::make_shared<DatasetTransform<TSeries>>(
        std::move(header),
        "default",
        std::move(series)
      );

      // Build and return the split
      return {DataSplit<TSeries>(split_name, std::move(rawd))};
    } catch (std::exception& e) {
      return {e.what()};
    }

  }

}