#include "pch.h"
#include <tempo/reader/new_reader.hpp>

#include "cli.hpp"

namespace fs = std::filesystem;

// For the given configuration (train set, test set, distance, parameters),
// this program looks for the k nearest neighbours of each train exemplar (within train but itself),
// and for each test exemplar (withing train).
// We also report ties, i.e. we can have several candidates as the first NN (or the second, or the third...).
//
// The results are compiled in a table for the train part, and another table for the test part.
// The row in the table correspond to the exemplar, in the order specified in the dataset
// (i.e. first exemplar is in the first row, etc...)
// The columns represent the NN: the first column store the 1st NN (maybe several on ties),
// the 2nd column store the 2nd NN (again, more than one).
// A candidate NN (which is always from the train set) is stored by its index,
// and its class (to ease the computation of the result), and the distance to the exemplar.
// Several indexes and classes can be stored on ties.


/// NN Cell structure for the table result
struct NNCell {
  std::vector<size_t> idxs{};
  std::vector<tempo::EL> classes{};
  double distance{tempo::utils::PINF};
};

/// Result table
struct ResultTable {

  std::vector<std::vector<NNCell>> table;

  ResultTable() = default;

  ResultTable(size_t nbexemplars, size_t k) :
    table(nbexemplars, std::vector<NNCell>(k)) {}

  std::vector<NNCell>& row(size_t exemplar_id) { return table[exemplar_id]; }

  NNCell& at(size_t exemplar_idx, size_t k_idx) { return table[exemplar_idx][k_idx]; }

};

/// Update a NNCell row
void update_row(size_t k, std::vector<NNCell>& row, size_t idx, tempo::EL el, double dist) {
  assert(row.size()==k);
  // Find insertion position
  size_t ipos = 0;
  for (; ipos<k&&row[ipos].distance<dist; ++ipos) {}
  // Insertion possible
  if (ipos<k) {
    NNCell& cell = row[ipos];
    // Ties
    if (cell.distance==dist) {
      cell.idxs.push_back(idx);
      cell.classes.push_back(el);
    } else {
      // Replace by insertion removal of the last
      row.insert(row.begin() + ipos, NNCell{{idx}, {el}, dist});
      row.pop_back();
    }
  }
  assert(row.size()==k);
}

/// NNk on the train set. Operate row by row: return a full row of the table
std::vector<NNCell> nnk_train(distfun_t& distance, tempo::DTS const& train_split, size_t self_idx, size_t k) {
  using namespace tempo;
  TSeries const& self = train_split[self_idx];
  // Index set: remove self from the list
  std::vector<size_t> is = train_split.index_set().vector();
  is.erase(is.begin() + self_idx);
  //
  double bsf = utils::PINF;
  std::vector<NNCell> results(k, NNCell());
  // Loop over other train exemplars
  for (const auto idx : is) {
    TSeries const& candidate = train_split[idx];
    EL el = train_split.label(idx).value();
    double dist = distance(self, candidate, bsf);
    if (dist<bsf) {
      update_row(k, results, idx, el, dist);
      bsf = results.back().distance;
    }
  }
  return results;
}

/// NNk on the test set. Operate row by row: return a full row of the table
std::vector<NNCell> nnk_test(distfun_t& distance, tempo::DTS const& train_split, tempo::TSeries const& self, size_t k) {
  using namespace tempo;
  double bsf = utils::PINF;
  std::vector<NNCell> results(k, NNCell());
  // Loop over train exemplars
  for (const auto idx : train_split.index_set()) {
    TSeries const& candidate = train_split[idx];
    EL el = train_split.label(idx).value();
    double dist = distance(self, candidate, bsf);
    if (dist<bsf) {
      update_row(k, results, idx, el, dist);
      bsf = results.back().distance;
    }
  }
  return results;
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;
  using namespace tempo::utils;


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
    if (p_out) { outpath = {fs::path{p_out.value()}}; }
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
  string dsname;
  {
    auto parg_UCRNAME = tempo::scli::get_parameter<string>(args, "-n", tempo::scli::extract_string);
    if (!parg_UCRNAME) { do_exit(1, "specify the UCR dataset name with the -n flag -n:NAME"); }
    dsname = parg_UCRNAME.value();
  }

  // distance
  dist_config dconf = cmd_dist(args);

  // --- --- --- --- --- ---
  // Read the dataset TRAIN.ts and TEST.ts files
  fs::path dspath = UCRPATH/dsname;

  DataSplit<TSeries> train_split = std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TRAIN.ts"), "train"));
  DatasetHeader const& train_header = train_split.header();
  size_t train_top = train_header.size();

  DataSplit<TSeries> test_split =
    std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TEST.ts"), "test", train_header.label_encoder()));
  DatasetHeader const& test_header = test_split.header();
  size_t test_top = test_header.size();

  // --- --- --- --- --- ---
  // Json Record
  Json::Value jv;

  jv["dataset"] = train_header.name();
  jv["distance"] = dconf.to_json();

  // --- --- --- --- --- ---
  // Sanity check
  {
    std::vector<std::string> errors = {};

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    auto [bcm, remainder] = train_split.get_BCM();

    if (!remainder.empty()) {
      errors.emplace_back("Train set: contains exemplar without label");
    }

    for (const auto& [label, vec] : bcm) {
      if (vec.size()<2) {
        errors.emplace_back("Train set: contains a class with only one exemplar");
        break;
      }
    }

    if (!errors.empty()) {
      jv["status"] = "error";
      jv["status_message"] = utils::cat(errors, "; ");
      cout << jv.toStyledString() << endl;
      if (outpath) {
        auto out = ofstream(outpath.value());
        out << jv << endl;
      }
      exit(2);
    }

  }


  // --- --- --- --- --- ---
  // Progress reporting
  utils::ProgressMonitor pm(train_top + test_top);  // How many to do, both train and test accuracy
  size_t nbdone = 0;                                // How many done up to "now"


  // --- --- --- --- --- ---
  // Train accuracy
  ResultTable train_table(train_top, k);
  {
    // Multithreading control
    utils::ParTasks ptasks;
    std::mutex mutex;

    // Task implemented over the train exemplars (self_idx = current train exemplar)
    auto task = [&](size_t self_idx) {
      std::vector<NNCell> results = nnk_train(dconf.dist_fun, train_split, self_idx, k);
      {
        std::lock_guard lock(mutex);
        nbdone++;
        pm.print_progress(cout, nbdone);
        train_table.row(self_idx) = move(results);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    p.execute(nbthreads, task, 0, train_top, 1);
  }


  // --- --- --- --- --- ---
  // Test accuracy
  ResultTable test_table(test_top, k);
  {
    // Multithreading control
    utils::ParTasks ptasks;
    std::mutex mutex;

    // Task implemented over the test exemplars (self_idx = current test exemplars)
    auto task = [&](size_t self_idx) {
      auto const& self = test_split[self_idx];
      std::vector<NNCell> results = nnk_test(dconf.dist_fun, train_split, self, k);
      {
        std::lock_guard lock(mutex);
        nbdone++;
        pm.print_progress(cout, nbdone);
        test_table.row(self_idx) = move(results);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    p.execute(nbthreads, task, 0, test_top, 1);
  }






  // --- --- --- --- --- ---
  // Display

  std::cout << "TRAIN TABLE";
  for (size_t i = 0; i<train_table.table.size(); ++i) {
    std::cout << std::endl << i << " cl " << train_split.label(i).value();

    for (const auto& cell : train_table.table[i]) {
      assert(cell.idxs.size()==cell.classes.size());
      std::cout << " | " << cell.distance << " ";
      for (size_t j = 0; j<cell.idxs.size(); ++j) {
        std::cout << "(" << cell.idxs[j] << ", " << cell.classes[j] << ") ";
      }
    }
  }
  std::cout << std::endl << std::endl;


  std::cout << "TEST TABLE";
  for (size_t i = 0; i<test_table.table.size(); ++i) {
    std::cout << std::endl << i << " cl " << test_split.label(i).value();

    for (const auto& cell : test_table.table[i]) {
      assert(cell.idxs.size()==cell.classes.size());
      std::cout << " | " << cell.distance << " ";
      for (size_t j = 0; j<cell.idxs.size(); ++j) {
        std::cout << "(" << cell.idxs[j] << ", " << cell.classes[j] << ") ";
      }
    }
  }
  std::cout << std::endl;





  /*

  Json::Value jtest;

  for (size_t qidx = 0; qidx<test_top; ++qidx) {
    auto const& vnn = kresults[qidx];
    Json::Value result_nnidx;
    Json::Value result_nnclass;
    Json::Value result_distance;
    for (auto const& nn : vnn) {
      result_nnidx.append(nn.candidate_nn_idx);
      result_nnclass.append(nn.candidate_nn_elabel);
      result_distance.append(nn.distance);
    }
    Json::Value res;
    res["idx"] = result_nnidx;
    res["class"] = result_nnclass;
    res["distance"] = result_distance;
    res["true_class"] = test_split.label(qidx).value();
    jtest.append(res);
  }

  jv["result_neighbour"] = jtest;

  cout << jv.toStyledString() << endl;

  */


  /*
  // PRNG used to break ties
  PRNG prng(seed + test_top*2 + 1);
  size_t bestnbcorrect = 0;
  double bestaccuracy = 0;
  size_t bestk = 0;

  for (size_t kk = 1; kk<=k; ++kk) {

    size_t nbcorrect = 0;

    for (size_t qidx = 0; qidx<test_top; ++qidx) {

      auto const& vnn = kresults[qidx];

      std::map<size_t, size_t> mapcount;
      std::map<size_t, double> mapdist;

      for (size_t ikk = 0; ikk<kk&&ikk<vnn.size(); ++ikk) {
        mapcount[vnn[ikk].candidate_nn_elabel] += 1;
        mapdist[vnn[ikk].candidate_nn_elabel] += vnn[ikk].distance;
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

      size_t true_class = test_split.label(qidx).value();

      if (selected_class==true_class) { nbcorrect++; }

    }

    double accuracy = (double)nbcorrect/double(test_top);
    if (nbcorrect>bestnbcorrect) {
      bestnbcorrect = nbcorrect;
      bestk = kk;
      bestaccuracy = accuracy;
    }

    cout << "k = " << kk << " nb correct = " << nbcorrect << "/" << test_top
         << " = " << accuracy << endl;
  }


  jv["status"] = "success";
  jv["accuracy"] = bestaccuracy;
  jv["k"] = bestk;
  jv["nbcorrect"] = bestnbcorrect;

   */

  cout << endl << jv.toStyledString() << endl;
  if (outpath) {
    auto out = ofstream(outpath.value());
    out << jv << endl;
  }

  return 0;
}
