#pragma once

#include <string>
#include <optional>
#include <variant>

#include <filesystem>
namespace fs = std::filesystem;

struct read_csv {
  fs::path train;
  fs::path test;
  std::string name;
  bool skip_header;
  char sep;
};

struct read_ucr {
  fs::path ucr_dir;
  std::string name;
  fs::path train;
  fs::path test;
};

struct cmdopt {
  std::variant<read_ucr, read_csv> input;
  size_t nb_trees;
  size_t nb_candidates;
  int nb_threads;
  std::string pfconfig;
  std::optional<fs::path> output;
};

std::variant<std::string, cmdopt> parse_cmd(int argc, char **argv);
