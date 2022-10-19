#pragma once

#include <tempo/reader/dts.reader.hpp>

#include <string>
#include <optional>
#include <variant>

#include <filesystem>
namespace fs = std::filesystem;
namespace trd = tempo::reader::dataset;

struct cmdopt {
  std::variant<trd::ts_ucr, trd::csv> input;
  size_t nb_trees;
  size_t nb_candidates;
  int nb_threads;
  std::string pfconfig;
  std::optional<fs::path> output;
  std::optional<fs::path> prob_output;
};

std::variant<std::string, cmdopt> parse_cmd(int argc, char **argv);
