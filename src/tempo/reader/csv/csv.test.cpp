#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <iostream>

#include "csv.hpp"

TEST_CASE("CSV reader") {

  std::string filename = "src/tempo/reader/csv/test.csv";
  std::ifstream input(filename, std::ios::binary);
  tempo::reader::CSVDataset result = tempo::reader::read_csv(input, false, ' ');

  // Check that we read the correct number lines
  REQUIRE(result.rows.size()==5);

  // Check the labels
  REQUIRE(result.labels.contains("label1"));
  REQUIRE(result.labels.contains("label2"));

  // Check the length
  REQUIRE(result.length_min == 5);
  REQUIRE(result.length_max == 10);

}
