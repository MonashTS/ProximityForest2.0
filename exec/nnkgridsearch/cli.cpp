
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
  "    -d:modminkowski:<float e>  Modified Minkowski distance with exponent 'e' (does not take the e-th root of the result)\n"
  "    -d:dtw:<float e>:<int w>               DTW with cost function exponent 'e' and warping window 'w'.\n"
  "                                           'w'<0 means no window\n"
  "    -d:adtw:<float e>:<float omega>        ADTW with cost function exponent 'e' and penalty 'omega'\n"
  "    -d:erp:<float e>:<float gv>:<int w>    ERP with cost function exponent 'e', gap value 'gv' and warping window 'w'\n"
  "                                           'w'<0 means no window\n"
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


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Optional args
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

void cmd_optional(std::vector<std::string> const& args, Config& conf) {

  // Value for k
  {
    auto p_k = tempo::scli::get_parameter<long long>(args, "-k", tempo::scli::extract_int, 1);
    if (p_k<=0) { do_exit(1, "-k must be followed by a integer >= 1"); }
    conf.k = p_k;
  }

  // Number of threads
  {
    auto p_et = tempo::scli::get_parameter<long long>(args, "-et", tempo::scli::extract_int, 0);
    if (p_et<0) { do_exit(1, "-et must specify a number of threads > 0, or 0 for auto-detect"); }
    if (p_et==0) { conf.nbthreads = std::thread::hardware_concurrency() + 2; } else { conf.nbthreads = p_et; }
  }

  // Random
  {
    auto p_seed = tempo::scli::get_parameter<long long>(args, "-seed", tempo::scli::extract_int, -1);
    if (p_seed<0) { conf.seed = std::random_device()(); } else { conf.seed = p_seed; }
    conf.pprng = std::make_unique<tempo::PRNG>(conf.seed);
  }

  // Output file
  {
    auto p_out = tempo::scli::get_parameter<std::string>(args, "-out", tempo::scli::extract_string);
    if (p_out) { conf.outpath = {std::filesystem::path{p_out.value()}}; }
  }
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Transform
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

/// Derivative -t:derivative:<degree>
bool t_derivative(std::vector<std::string> const& v, Config& conf) {
  using namespace std;
  using namespace tempo;

  if (v[0]=="derivative") {
    bool ok = v.size()==2;
    if (ok) {
      auto od = tempo::reader::as_int(v[1]);
      ok = od.has_value();
      if (ok) {

        int degree = od.value();
        conf.param_derivative_degree = {degree};

        auto train_derive_t = std::make_shared<DatasetTransform<TSeries>>(
          std::move(tempo::transform::derive(conf.loaded_train_split.transform(), degree).back())
        );
        conf.train_split = DTS("train", train_derive_t);

        auto test_derive_t = std::make_shared<DatasetTransform<TSeries>>(
          std::move(tempo::transform::derive(conf.loaded_test_split.transform(), degree).back())
        );

        conf.test_split = DTS("test", test_derive_t);
      }
    }
    // Catchall
    if (!ok) { do_exit(1, "DTW parameter error"); }
    return true;
  }
  return false;
}

/// Command line parsing: special helper for the configuration of the transform
void cmd_transform(std::vector<std::string> const& args, Config& conf) {
  using namespace std;
  using namespace tempo;

  // Optional -t flag
  auto parg_transform = tempo::scli::get_parameter<string>(args, "-t", tempo::scli::extract_string);
  if (parg_transform) {
    // Split on ':'
    std::vector<std::string> v = tempo::reader::split(parg_transform.value(), ':');
    conf.transform_name = v[0];
    // --- --- --- --- --- ---
    // Try parsing distance argument
    if (t_derivative(v, conf)) {}
      // --- --- --- --- --- ---
      // Unknown transform
    else { do_exit(1, "Unknown transform '" + v[0] + "'"); }

  } else {
    // Default transform
    conf.transform_name = conf.loaded_train_split.get_transform_name();
    conf.train_split = conf.loaded_train_split;
    conf.test_split = conf.loaded_test_split;
  }

}



// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// DISTANCE
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

/// Minkowski -d:minkowski:<e>
bool d_minkowski(std::vector<std::string> const& v, Config& conf) {
  using namespace std;
  using namespace tempo;
  if (v[0]=="modminkowski") {
    bool ok = v.size()==2;
    if (ok) {
      auto oe = tempo::reader::as_double(v[1]);
      ok = oe.has_value();
      if (ok) {
        // Extract params
        double param_cf_exponent = oe.value();
        // Create the distance
        conf.dist_fun = [=](TSeries const& A, TSeries const& B, double /* ub */) -> double {
          return distance::minkowski(A, B, param_cf_exponent);
        };
        // Record params
        conf.param_cf_exponent = {param_cf_exponent};
      }
    }
    // Catchall
    if (!ok) { do_exit(1, "Minkowski parameter error"); }
    return true;
  }
  return false;
}

/// DTW -d:dtw:<e>:<w>
bool d_dtw(std::vector<std::string> const& v, Config& conf) {
  using namespace std;
  using namespace tempo;

  if (v[0]=="dtw") {
    bool ok = v.size()==3;
    if (ok) {
      auto oe = tempo::reader::as_double(v[1]);
      optional<long> ow = tempo::reader::as_int(v[2]);
      ok = oe.has_value()&&ow.has_value();
      if (ok) {
        // Extract params
        double param_cf_exponent = oe.value();
        size_t param_window = utils::NO_WINDOW;
        if (ow.value()>=0) { param_window = ow.value(); }
        // Create the distance
        conf.dist_fun = [=](TSeries const& A, TSeries const& B, double ub) -> double {
          return distance::dtw(
            A.size(),
            B.size(),
            distance::univariate::ade<TSeries>(param_cf_exponent)(A, B),
            param_window,
            ub
          );
        };
        // Record params
        conf.param_cf_exponent = {param_cf_exponent};
        conf.param_window = {-1};
        if (ow.value()>=0) { conf.param_window = ow.value(); }
      }
    }
    // Catchall
    if (!ok) { do_exit(1, "DTW parameter error"); }
    return true;
  }

  return false;
}

/// ADTW -d:adtw:<e>:<omega>
bool d_adtw(std::vector<std::string> const& v, Config& conf) {
  using namespace std;
  using namespace tempo;

  if (v[0]=="adtw") {
    bool ok = v.size()==3;
    if (ok) {
      auto oe = tempo::reader::as_double(v[1]);
      auto oo = tempo::reader::as_double(v[2]);
      ok = oe.has_value()&&oo.has_value();
      if (ok) {
        // Extract params
        double param_cf_exponent = oe.value();
        double param_omega = oo.value();
        // Create the distance
        conf.dist_fun = [=](TSeries const& A, TSeries const& B, double ub) -> double {
          return distance::adtw(
            A.size(),
            B.size(),
            distance::univariate::ade<TSeries>(param_cf_exponent)(A, B),
            param_omega,
            ub
          );
        };
        // Record params
        conf.param_cf_exponent = {param_cf_exponent};
        conf.param_omega = param_omega;
      }
    }
    // Catchall
    if (!ok) { do_exit(1, "ADTW parameter error"); }
    return true;
  }
  return false;
}

/// ERP -d:erp:<e>:<gv>:<w>
bool d_erp(std::vector<std::string> const& v, Config& conf) {
  using namespace std;
  using namespace tempo;

  if (v[0]=="erp") {
    bool ok = v.size()==4;
    if (ok) {
      optional<double> oe = tempo::reader::as_double(v[1]);
      optional<double> ogv = tempo::reader::as_double(v[2]);
      optional<long> ow = tempo::reader::as_int(v[3]);
      ok = oe.has_value()&&ogv.has_value()&&ow.has_value();
      if (ok) {
        // Extract params
        double param_cf_exponent = oe.value();
        double param_gv = ogv.value();
        size_t param_window = utils::NO_WINDOW;
        if (ow.value()>=0) { param_window = ow.value(); }
        // Create the distance
        conf.dist_fun = [=](TSeries const& A, TSeries const& B, double ub) -> double {
          return distance::erp(
            A.size(),
            B.size(),
            tempo::distance::univariate::adegv<TSeries>(param_cf_exponent)(A, param_gv),
            tempo::distance::univariate::adegv<TSeries>(param_cf_exponent)(B, param_gv),
            distance::univariate::ade<TSeries>(param_cf_exponent)(A, B),
            param_window,
            ub
          );
        };
        // Record params
        conf.param_cf_exponent = {param_cf_exponent};
        conf.param_gap_value = {param_gv};
        conf.param_window = {-1};
        if (ow.value()>=0) { conf.param_window = ow.value(); }
      }
    }
    // Catchall
    if (!ok) { do_exit(1, "ADTW parameter error"); }
    return true;
  }
  return false;

}

/// Command line parsing: special helper for the configuration of the distance
void cmd_dist(std::vector<std::string> const& args, Config& conf) {
  using namespace std;
  using namespace tempo;

  // We must find a '-d' flag, else error
  auto parg_dist = tempo::scli::get_parameter<string>(args, "-d", tempo::scli::extract_string);
  if (!parg_dist) { do_exit(1, "specify a distance to use with '-d'"); }

  // Split on ':'
  std::vector<std::string> v = tempo::reader::split(parg_dist.value(), ':');
  conf.dist_name = v[0];

  // --- --- --- --- --- ---
  // Try parsing distance argument
  if (d_minkowski(v, conf)) {}
  else if (d_dtw(v, conf)) {}
  else if (d_adtw(v, conf)) {}
  else if (d_erp(v, conf)) {}
    // --- --- --- --- --- ---
    // Unknown distance
  else { do_exit(1, "Unknown distance '" + v[0] + "'"); }

}
