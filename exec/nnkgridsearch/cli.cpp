
#include "cli.hpp"

std::string usage =
  "Time Series NNK Classification - demonstration application\n"
  "Monash University, Melbourne, Australia 2022\n"
  "Dr. Matthieu Herrmann\n"
  "This application works with the UCR archive using the TS file format (or any archive following the same conventions).\n"
  "For each exemplar in '_TEST', search for the k nearest neighbours in _'TRAIN' and report the results as json.\n"
  "Ties are broken randomly.\n"
  "nnk <-p:> <-n:> <-d:> [-k:] [-et:] [-seed:]\n"
  "Mandatory arguments:\n"
  "  -p:<path to the ucr archive folder>   e.g. '-p:/home/myuser/Univariate_ts'\n"
  "  -n:<name of the dataset>              e.g. '-n:Adiac' Must correspond to the dataset's folder name\n"
  "  -d:<distance>\n"
  "    -d:minkowski:<float e>     Minkowski distance with exponent 'e'\n"
  "    -d:dtw:<float e>:<int w>   DTW with cost function exponent e and warping window w. w<0 means no window\n"
  "Optional arguments [with their default values]:\n"
  "  -et:<int n>     Number of execution threads. Autodetect if n=<0 [n = 0]\n"
  "  -k:<int n>      Number of neighbours to search [n = 1])\n"
  "  -seed:<int n>   Fixed seed of randomness. Generate a random seed if n<0 [n = -1] !\n"
  "  -out:<path>     Where to write the json file. If the file exists, overwrite it."
  "";

[[noreturn]] void do_exit(int code, std::optional<std::string> msg) {
  if (code==0) {
    if (msg) { std::cout << msg.value() << std::endl; }
  } else {
    std::cerr << usage << std::endl;
    if (msg) { std::cerr << msg.value() << std::endl; }
  }
  exit(code);
}


/// Command line parsing: special helper for the distance configuration
dist_config cmd_dist(std::vector<std::string> const& args) {
  using namespace std;
  using namespace tempo;

  dist_config dconf;

  static string minkowski = "minkoski";
  static string dtw = "dtw";

  // We must find a '-d' flag, else error
  auto parg_dist = tempo::scli::get_parameter<string>(args, "-d", tempo::scli::extract_string);
  if (!parg_dist) { do_exit(1, "specify a distance to use with '-d'"); }

  // Split on ':'
  auto v = tempo::reader::split(parg_dist.value(), ':');

  dconf.dist_name = v[0];

  // --- --- --- --- --- ---
  // Minkowski -d:minkowski:<e>
  if (v[0]==minkowski) {
    bool ok = v.size()==2;
    if (ok) {
      auto oe = tempo::reader::as_double(v[1]);
      ok = oe.has_value();
      if (ok) {
        // Create the distance
        double param_cf_exponent = oe.value();
        dconf.dist_fun = [=](TSeries const& A, TSeries const& B, double /* ub */) -> double {
          return distance::minkowski(A, B, param_cf_exponent);
        };
        // Record param
        dconf.param_cf_exponent = {param_cf_exponent};
      }
    }
    // Catchall
    if (!ok) { do_exit(1, "Minkowski parameter error"); }
  }
    // --- --- --- --- --- ---
    // DTW -d:dtw:<e>:<w>
  else if (v[0]==dtw) {
    bool ok = v.size()==3;
    if (ok) {
      auto oe = tempo::reader::as_double(v[1]);
      optional<long> ow = tempo::reader::as_int(v[2]);
      ok = oe.has_value()&&ow.has_value();
      if (ok) {
        // Create the distance
        double param_cf_exponent = oe.value();
        size_t param_window = utils::NO_WINDOW;
        if (ow.value()>=0) { param_window = ow.value(); }
        dconf.dist_fun = [=](TSeries const& A, TSeries const& B, double ub) -> double {
          return distance::dtw(
            A.size(),
            B.size(),
            distance::univariate::ade<TSeries>(param_cf_exponent)(A, B),
            param_window,
            ub
          );
        };
        // Record param
        dconf.param_cf_exponent = {param_cf_exponent};
        dconf.param_window = {-1};
        if (ow.value()>=0) { dconf.param_window = ow.value(); }
      }
    }
    // Catchall
    if (!ok) { do_exit(1, "DTW parameter error"); }
  }
    // --- --- --- --- --- ---
    // Unknown distance
  else { do_exit(1, "Unknown distance '" + v[0] + "'"); }

  return dconf;
}
