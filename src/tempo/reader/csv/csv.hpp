#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <set>

namespace tempo::reader {

  struct CSVRow {
    std::string label;
    std::vector<double> data;

    CSVRow() : label(), data() {}

    CSVRow(std::string label, std::vector<double>&& d) : label(std::move(label)), data(std::move(d)) {}

    // Do not copy me!
    CSVRow(CSVRow const& other) = delete;
    CSVRow& operator =(CSVRow const& other) = delete;

    // Move me!
    CSVRow(CSVRow&& other) = default;
    CSVRow& operator =(CSVRow&& other) = default;
  };

  struct CSVDataset {
    std::set<std::string> labels;
    std::vector<CSVRow> rows;
    size_t length_min;
    size_t length_max;
  };

  CSVDataset read_csv(std::istream& input, bool header, char sep);

} // End of namespace tempo::reader

