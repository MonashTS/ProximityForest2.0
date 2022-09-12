#include "cmdline.hpp"

#include <tclap/CmdLine.h>
#include <cassert>
#include <string>
#include <vector>
#include <thread>
#include <set>
#include <regex>

std::variant<std::string, cmdopt> parse_cmd(int argc, char **argv) {
  using namespace std;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Command line parsing
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  try {

    // --- --- --- Build the cmd parser

    // Command line object with a "Command description message", delimiter (usually space), and the version number.
    // The CmdLine object parses the argv array based on the Arg objects that it contains.
    TCLAP::CmdLine cmd("Splitting Forest", ' ', "0.0.1");

    // --- Dataset UCR or CSV
    TCLAP::SwitchArg ucr("", "ucr", "use a UCR dataset", false);
    TCLAP::SwitchArg csv("", "csv", "use a CSV dataset", false);
    cmd.xorAdd(ucr, csv);
    TCLAP::UnlabeledMultiArg<string> ds("dataset", "<ucr_path ucr_name> or <csv_train csv_test>", true, "strings", cmd);

    // --- Extra CSV
    TCLAP::SwitchArg csv_skip("", "csv-skip-header", "Skip the csv's first line", cmd, false);
    TCLAP::ValueArg<std::string> csv_sep("", "csv-separator", "CSV columns separator", false, ",", "character", cmd);

    // --- PF version
    TCLAP::ValueArg<string> pfconfig("", "pfc", "PF Configuration", false, "pf2018", "PF Configuration", cmd);

    // --- Tree config
    TCLAP::ValueArg<int> nbt("t", "nb-trees", "Number of trees", true, 100, "int", cmd);
    TCLAP::ValueArg<int> nbc("c", "nb-candidates", "Number of candidates", true, 5, "int", cmd);

    // --- Parallelism
    TCLAP::ValueArg<int> nbp("p", "nb-threads", "Number of threads - use <=0 for autodetect", false, 1, "int", cmd);

    // --- Output
    TCLAP::ValueArg<string> out("o", "out", "path to output json file", false, "", "string", cmd);

    // --- --- --- Parse the argv array.
    cmd.parse(argc, argv);

    // --- --- --- Get options
    cmdopt opt{};

    std::vector<std::string> remainder = ds.getValue();

    // --- Input
    if (ucr.isSet()) {
      // --- --- --- UCR
      read_ucr ru{};
      if (remainder.size()<2) { return {"Expects ucr <path to ucr> <ucr dataset name>"}; }
      ru.ucr_dir = fs::path(remainder[0]);
      ru.name = remainder[1];
      remainder.erase(remainder.begin(), remainder.begin()+1);
      // --- --- ---
      ru.train = ru.ucr_dir/ru.name/(ru.name + "_TRAIN.ts");
      ru.test = ru.ucr_dir/ru.name/(ru.name + "_TEST.ts");
      if (!fs::exists(ru.train)) { return {"UCR Train file " + ru.train.string() + " not found."}; }
      if (!fs::exists(ru.test)) { return {"UCR Test file " + ru.test.string() + " not found."}; }
      opt.input = {ru};
    } else {
      // --- --- --- CSV
      assert(csv.isSet());
      read_csv rc{};
      if (remainder.size()<3) { return {"Expects csv <train path> <test path> <dataset name>"}; }
      rc.train = fs::path(remainder[0]);
      rc.test = fs::path(remainder[1]);
      rc.name = fs::path(remainder[2]);
      remainder.erase(remainder.begin(), remainder.begin()+3);
      rc.skip_header = csv_skip.getValue();
      rc.sep = csv_sep.getValue().at(0);
      // --- --- ---
      if (!fs::exists(rc.train)) { return {"CSV Train file " + rc.train.string() + " not found."}; }
      if (!fs::exists(rc.test)) { return {"CSV Test file " + rc.test.string() + " not found."}; }
      opt.input = {rc};
    }

    // --- PF Config
    opt.pfconfig = pfconfig.getValue();

    /*
    set<string> allowed{"pf2018"};
    if (!allowed.contains(opt.pfconfig)) {
      regex r(":");
      auto const& str = opt.pfconfig;
      vector<string> split(sregex_token_iterator(str.begin(), str.end(), r, -1), sregex_token_iterator());
      if(split[0]=="custom"){ return {"custom not implemented"}; }
      else { return {"Unrecognized configuration '" + str + "'"}; }
    }
     */

    // --- Other options
    opt.nb_trees = nbt.getValue();
    opt.nb_candidates = nbc.getValue();
    opt.nb_threads = nbp.getValue()<=0 ? std::thread::hardware_concurrency() : nbp.getValue();
    if(out.isSet()){ opt.output = {out.getValue()}; }

    return {opt};

  } catch (TCLAP::ArgException& e)  // catch exceptions
  { return {std::string("error: " + e.error() + " for arg " + e.argId())}; }
}
