#include "csv.hpp"
#include <rapidcsv.h>

namespace tempo::reader {

  CSVDataset read_csv(std::istream& input, bool header, char sep) {

    // Configure header
    rapidcsv::LabelParams label_params;
    if (!header) { label_params.mColumnNameIdx = -1; }

    // Prepare reader
    rapidcsv::Document doc(input,
                           label_params,
                           rapidcsv::SeparatorParams(sep)
    );

    size_t nbrow = doc.GetRowCount();

    std::vector<CSVRow> rows;
    rows.reserve(nbrow);
    size_t min_length = std::numeric_limits<size_t>::max();
    size_t max_length = 0;

    std::set<std::string> labels{};

    for (size_t i = 0; i<nbrow; ++i) {
      // Read the line
      std::vector<std::string> row = doc.GetRow<std::string>(i);
      // Get info
      size_t row_length = row.size()-1; // Minus 1 because the label starts the row
      min_length = std::min(min_length, row_length);
      max_length = std::max(max_length, row_length);
      // Extract label and data
      std::string label = row.front();
      std::vector<double> data(row.size() - 1);
      std::transform(row.begin() + 1, row.end(), data.begin(), [](const std::string& val) { return std::stod(val); });
      // Update label set
      labels.insert(label);
      // Create the row
      rows.emplace_back(std::move(label), std::move(data));
    }

    return CSVDataset{std::move(labels), std::move(rows), min_length, max_length};

  }

}