#include "pch.h"
#include <tempo/utils/simplecli.hpp>
#include <tempo/distance/lockstep/minkowski.hpp>

namespace fs = std::filesystem;

static std::string usage =
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

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (code==0) {
    if (msg) { std::cout << msg.value() << std::endl; }
  } else {
    std::cerr << usage << std::endl;
    if (msg) { std::cerr << msg.value() << std::endl; }
  }
  exit(code);
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;
  using distfun_t = std::function<double(TSeries const& A, TSeries const& B, double ub)>;


  // --- --- --- --- --- ---
  // Program Arguments
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if (args.empty()) { do_exit(0, usage); }

  // --- --- ---
  // Optional

  // Value for k
  size_t k;
  {
    auto p_k = tempo::scli::get_parameter<long long>(args, "-k", tempo::scli::extract_int, 1);
    if (p_k<=0) { do_exit(1, "-k must be followed by a integer >= 1"); }
    k = p_k;
  }

  // Number of threads
  size_t nbthreads;
  {
    auto p_et = tempo::scli::get_parameter<long long>(args, "-et", tempo::scli::extract_int, 0);
    if (p_et<0) { do_exit(1, "-et must specify a number of threads > 0, or 0 for auto-detect"); }
    if (p_et==0) { nbthreads = std::thread::hardware_concurrency() + 2; } else { nbthreads = p_et; }
  }

  // Random
  size_t seed;
  {
    auto p_seed = tempo::scli::get_parameter<long long>(args, "-seed", tempo::scli::extract_int, 0);
    if (p_seed<0) { seed = std::random_device()(); } else { seed = p_seed; }
  }

  // Output file
  optional<fs::path> outpath{};
  {
    auto p_out = tempo::scli::get_parameter<string>(args, "-out", tempo::scli::extract_string);
    if(p_out){ outpath = {fs::path{p_out.value()}}; }
  }

  // --- --- ---
  // Mandatory

  // Path to UCR
  fs::path UCRPATH;
  {
    auto parg_UCRPATH = tempo::scli::get_parameter<string>(args, "-p", tempo::scli::extract_string);
    if (!parg_UCRPATH) { do_exit(1, "specify the UCR path with the -p flag, e.g. -p:/path/to/dataset"); }
    UCRPATH = fs::path(parg_UCRPATH.value());
  }

  // Dataset name
  string dataset_name;
  {
    auto parg_UCRNAME = tempo::scli::get_parameter<string>(args, "-n", tempo::scli::extract_string);
    if (!parg_UCRNAME) { do_exit(1, "specify the UCR dataset name with the -n flag -n:NAME"); }
    dataset_name = parg_UCRNAME.value();
  }

  // distance
  double param_cf_exponent;
  size_t param_window;
  distfun_t distfun;
  {
    auto parg_dist = tempo::scli::get_parameter<string>(args, "-d", tempo::scli::extract_string);
    if (!parg_dist) { do_exit(1, "specify a distance to use with '-d'"); }
    auto v = tempo::reader::split(parg_dist.value(), ':');
    // Minkowski
    if (v[0]=="minkowski") {
      bool ok = v.size()==2;
      if (ok) {
        auto oe = tempo::reader::as_double(v[1]);
        ok = oe.has_value();
        if (ok) {
          param_cf_exponent = oe.value();
          distfun = [=](TSeries const& A, TSeries const& B, double /* ub */) -> double {
            return distance::minkowski(A, B, param_cf_exponent);
          };
        }
      }
      // Catchall
      if (!ok) { do_exit(1, "Minkowski parameter error"); }
    } // DTW with or without window
    else if (v[0]=="dtw") {
      bool ok = v.size()==3;
      if (ok) {
        auto oe = tempo::reader::as_double(v[1]);
        auto ow = tempo::reader::as_int(v[2]);
        ok = oe.has_value()&&ow.has_value();
        if (ok) {
          param_window = ow.value()<0 ? utils::NO_WINDOW : ow.value();
          param_cf_exponent = oe.value();
          distfun = [=](TSeries const& A, TSeries const& B, double ub) -> double {
            return distance::dtw(
              A.size(), B.size(),
              distance::univariate::ade<F, TSeries>(param_cf_exponent)(A, B),
              param_window,
              ub
            );
          };
        }
      }
      // Catchall
      if (!ok) { do_exit(1, "DTW parameter error"); }
    } // Unknown distance
    else { do_exit(1, "Unknown distance '" + v[0] + "'"); }
  }

  // --- --- ---
  // Json Record
  Json::Value j;

  // Load UCR train and test dataset - Load test with the label encoder from train!
  auto variant_ds = reader::load_ucr_ts(UCRPATH, dataset_name);
  if (variant_ds.index()==0) { do_exit(2, {get<0>(variant_ds)}); }
  reader::Loaded_UCRDataset ds = get<1>(variant_ds);
  j["dataset"] = ds.to_json();

  // --- --- --- --- --- ---
  // Shorthands
  DTS const& train_dataset = ds.train_dataset;
  const auto& train_header = train_dataset.header();
  const size_t train_top = train_header.size();

  DTS const& test_dataset = ds.test_dataset;
  const auto& test_header = test_dataset.header();
  const size_t test_top = test_header.size();

  // --- --- --- --- --- ---
  // NN struct for the table 'kresults'
  // Each line in the table corresponds to a test exemplar.
  // Each column corresponds to a neighbor: 1st column is the nearest, follow by the one in the second column, etc...
  // Each cell in the table is of the following type: storing the idx of the candidate (in the train set), its class,
  // and the resulting distance.
  // Insertion strategy, and how to deal with ties (how to select which one to keep?)
  // We insert a candidate in the table (in a column) if:
  //   - we have less than k column (no ties to manage)
  //   - if we have k column and the current distance is =< than the one stored in the last column,
  //     insert the new record in the right position in the line, removing the last cell (previous kth NN)
  // This means that, on ties, the last encountered one "wins" (stays in the table).
  // To avoid any bias due to the ordering of the train set, we take candidate in a random order
  struct nn {
    size_t candidate_nn_idx;
    size_t candidate_nn_class;
    double distance;
  };

  vector<vector<nn>> kresults(test_top);

  // --- --- --- --- --- ---
  // Multithreading control
  utils::ParTasks ptasks;
  std::mutex mutex;

  // --- --- --- --- --- ---
  // Progress reporting
  utils::ProgressMonitor pm(test_top);  // How many to do
  size_t nbdone = 0;                    // How many done up to "now"

  // --- --- --- --- --- ---
  // Implement the task with an 'increment task', i.e. a task taking, when executed, a unique index as its parameter
  auto test_task = [&](size_t qidx) {
    size_t local_seed = seed+qidx;
    TSeries const& query = test_dataset[qidx];
    double bsf = tempo::utils::PINF<double>; // bsf = worst of the knn
    std::vector<nn> results;
    results.reserve(k);
    // Take candidates in random order
    std::vector<size_t> candidate_idxs(train_top);
    std::iota(candidate_idxs.begin(), candidate_idxs.end(), 0);
    std::shuffle(candidate_idxs.begin(), candidate_idxs.end(), std::mt19937{local_seed});
    // Candidate loop
    for (size_t candidateidx : candidate_idxs) {
      TSeries const& candidate = train_dataset[candidateidx];
      double dist = distfun(query, candidate, bsf);
      // Update knn
      if (dist<=bsf) {
        // Find position and insert
        size_t i = 0;
        for (; i<results.size()&&dist>results[i].distance; ++i) {}
        nn n{candidateidx, train_header.label_index(candidateidx).value(), dist};
        results.insert(results.begin() + i, n);
        // Remove last neighbour if we have too many candidates
        if (results.size()>k) { results.pop_back(); }
        // Update bsf
        bsf = results.back().distance;
      }
    }
    {
      std::lock_guard lock(mutex);
      nbdone++;
      pm.print_progress(cout, nbdone);
      // Write results
      kresults[qidx] = move(results);
    }
  };

  // Create the tasks per tree. Note that we clone the state.
  tempo::utils::ParTasks p;

  p.execute(nbthreads, test_task, 0, test_top, 1);

  Json::Value jtest;

  for (size_t qidx = 0; qidx<test_top; ++qidx) {
    auto const& vnn = kresults[qidx];
    Json::Value result_nnidx;
    Json::Value result_nnclass;
    Json::Value result_distance;
    for (auto const& nn : vnn) {
      result_nnidx.append(nn.candidate_nn_idx);
      result_nnclass.append(nn.candidate_nn_class);
      result_distance.append(nn.distance);
    }
    Json::Value res;
    res["idx"] = result_nnidx;
    res["class"] = result_nnclass;
    res["distance"] = result_distance;
    res["true_class"] = test_header.label_index(qidx).value();
    jtest.append(res);
  }

  j["result_neighbour"] = jtest;

  cout << j.toStyledString() << endl;



  // PRNG used to break ties
  PRNG prng(seed+test_top*2+1);
  for (size_t kk = 1; kk<=k; ++kk) {

    size_t nbcorrect = 0;

    for (size_t qidx = 0; qidx<test_top; ++qidx) {

      auto const& vnn = kresults[qidx];

      std::map<size_t, size_t> mapcount;
      std::map<size_t, double> mapdist;

      for (size_t ikk = 0; ikk<kk; ++ikk) {
        mapcount[vnn[ikk].candidate_nn_class] += 1;
        mapdist[vnn[ikk].candidate_nn_class] += vnn[ikk].distance;
      }

      size_t maxcount = 0;
      std::vector<size_t> maxclass;

      for (auto const& [cl, count] : mapcount) {
        if (count>maxcount) {
          maxcount = count;
          maxclass.clear();
          maxclass.push_back(cl);
        } else if (count==maxcount) {
          maxclass.push_back(cl);
        }
      }

      vector<size_t> closest_class;
      double sumdist = utils::PINF<double>;
      for (auto const& mc : maxclass) {
        if (mapdist[mc]<sumdist) {
          sumdist = mapdist[mc];
          closest_class.clear();
          closest_class.push_back(mc);
        } else if (mapdist[mc]==sumdist) {
          closest_class.push_back(mc);
        }
      }




      size_t selected_class = utils::pick_one(closest_class, prng);

      size_t true_class = test_header.label_index(qidx).value();

      if (selected_class==true_class) { nbcorrect++; }

    }

    cout << "k = " << kk << " nb correct = " << nbcorrect << "/" << test_top
         << " = " << (double)nbcorrect/double(test_top) << endl;

  }

  return 0;
}
