#pragma once

#include <vector>
#include <string>
#include <functional>
#include <set>
#include <algorithm>
#include <map>
#include <any>
#include <optional>

namespace tempo::scli {

  /// Simple CLI parser - core class
  class SCLICore {

  };



  /// ArgReader: given a vector of arguments, update the result map and returns the set of read index
  using ArgReader = std::function<std::set<size_t>(
    std::vector<std::string> const& args,
    std::map<std::string, std::any>& to_update
  )>;

  /// Simple CLI parser main class
  class SCLI {

    // Result map
    std::map<std::string, std::any> result;

    // Set of "unparse" index
    std::set<size_t> unparsed;

    // Set of reading functions
    std::vector<ArgReader> readers;

    template<typename T>
    class Getter {
      std::string key;

      explicit Getter(std::string k) : key(std::move(k)) {}

      T at(SCLI const& scli) override {
        return std::any_cast<T>(scli.result.at(key));
      }
    };

    template<typename T>
    Getter<T> register_reader(std::string k) {

    }

    inline std::map<std::string, std::any> parse(std::vector<std::string> args) {
      // Reset result map
      result.clear();

      // Prepare set of index for unparsed args
      unparsed.clear();
      for (size_t i = 0; i<args.size(); ++i) { unparsed.insert(i); }

      // Run each reader on the arguments
      for (auto const& r : readers) {
        std::set<size_t> s = r(args, result);
        std::set<size_t> inter;
        std::set_difference(unparsed.begin(), unparsed.end(), s.begin(), s.end(),
                            std::inserter(inter, inter.begin()));
        unparsed = inter;
      }
    }

  };








  /// Helper type: reading function. Try to read a string
  template<typename T>
  using fread_t = std::function<std::optional<T>(std::string const&)>;

  /// Helper type: prefix extracting function.
  template<typename T>
  using fpfxext_t = std::function<std::optional<T>(std::string const&, std::string const&)>;

  /// Get a parameter with a fread function. Return the default value 'defval' if not found.
  template<typename T>
  T get_parameter(std::vector<std::string> const& argv, fread_t<T> read, T defval) {
    for (auto const& a : argv) {
      auto oa = read(a);
      if (oa) { return oa.value(); }
    }
    return defval;
  }

  /// Get a parameter with a fread function. Return an empty option if not found.
  template<typename T>
  std::optional<T> get_parameter(std::vector<std::string> const& argv, fread_t<T> read) {
    for (auto const& a : argv) {
      auto oa = read(a);
      if (oa) { return oa; }
    }
    return {};
  }

  /// Read several parameters with a fread function. Return an empty vector if none found.
  template<typename T>
  std::vector<T> get_parameters(std::vector<std::string> const& argv, fread_t<T> read) {
    std::vector<T> result;
    for (auto const& a : argv) {
      auto oa = read(a);
      if (oa) { result.push_back(oa.value()); }
    }
    return result;
  }

  // --- --- ---
  // --- --- --- fread helpers B: look for flag, extract data separated with ':'  (e.g. -n:5)
  // --- --- ---

  /// Check for a switch
  inline bool get_switch(std::vector<std::string> const& args, std::string pfx) {
    return get_parameter<bool>(
      args,
      [=](std::string const& a) -> std::optional<bool> { if (a==pfx) { return {true}; } else { return {}; }},
      false
    );
  }

  /// Check for an optional parameter
  template<typename T>
  inline std::optional<T> get_parameter(std::vector<std::string> const& args, std::string pfx, fpfxext_t<T> f) {
    pfx = pfx + ':';
    return get_parameter<T>(args, [=](std::string const& a) { return f(a, pfx); });
  }

  /// Check for a parameter with a default value
  template<typename T>
  inline T get_parameter(std::vector<std::string> const& args, std::string pfx, fpfxext_t<T> f, T def) {
    pfx = pfx + ':';
    return get_parameter<T>(args, [=](std::string const& a) { return f(a, pfx); }, def);
  }

} // End of namespace SCLI
