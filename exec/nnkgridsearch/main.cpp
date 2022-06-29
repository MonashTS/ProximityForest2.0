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
// the 2nd column store the 2nd NN (again, possibly more than one).
// A candidate NN (which is always from the train set) is stored by its index,
// and its class (to ease the computation of the result), and the distance to the exemplar.
// Several indexes and classes can be stored on ties.
//
// This setting actually records the k nearest "distance values", and for each "distance value", it stores
// the corresponding train exemplars. Hence, we can have e.g. k=2 and have in the table 5 exemplar:
// | dv=5, ex1, ex2, ex3 | dv = 6, ex4, ex5 |
// In a "usual" NN, we would only have a subset of {ex1, ex2, ex3}.


/// NN Cell structure for the table result
struct NNCell {
  std::vector<std::tuple<size_t, tempo::EL>> idx_label_v;
  double distance{tempo::utils::PINF};
};

/// Result table
struct ResultTable {

  std::vector<std::vector<NNCell>> table;

  ResultTable() = default;

  ResultTable(size_t nbexemplars, size_t k) :
    table(nbexemplars, std::vector<NNCell>(k)) {}

  std::vector<NNCell>& row(size_t idx) { return table[idx]; }

  std::vector<NNCell>& operator [](size_t idx) { return table[idx]; }

  std::vector<NNCell> const& operator [](size_t idx) const { return table[idx]; }

  NNCell& at(size_t exemplar_idx, size_t k_idx) { return table[exemplar_idx][k_idx]; }

  Json::Value to_json(std::function<tempo::EL(size_t)> get_label) {
    Json::Value results = Json::arrayValue;
    for (size_t i = 0; i<table.size(); ++i) {
      // For each row in the table, store the idx and the actual label
      Json::Value full_row = Json::objectValue;
      full_row["idx"] = (int)i;
      full_row["label"] = get_label(i);
      // Create the cells of the table (list of nearest neighbours grouped by increasing distances)
      Json::Value columns = Json::arrayValue;
      for (const auto& cell : table[i]) {
        // For each cell, collect all candidates and associated label
        Json::Value candidates = Json::arrayValue;
        Json::Value labels = Json::arrayValue;
        for (auto [idx, label] : cell.idx_label_v) {
          candidates.append(idx);
          labels.append(label);
        }
        // Create the cell object
        Json::Value jscell = Json::objectValue;
        jscell["distance"] = cell.distance;
        jscell["candidates"] = candidates;
        jscell["labels"] = labels;
        // add the cell to the columns
        columns.append(jscell);
      }
      // Add the columns (cells) to the row
      full_row["nn"] = columns;
      // Add the row to the table
      results.append(full_row);
    }

    return results;
  }

};

/// Compute the number of correct classification for a given k with majority vote and random cutting of ties
size_t nb_correct_01loss(size_t const k, ResultTable const& table, tempo::DTS const& split, tempo::PRNG& prng) {
  size_t nbcorrect = 0;
  assert(table.table.size()==split.size());

  for (size_t i = 0; i<split.size(); ++i) {
    std::vector<NNCell> const& row = table[i];

    // Count votes and cumulative distance per label
    std::map<tempo::EL, size_t> votes;
    std::map<tempo::EL, double> votes_dist;
    int remaining_k = (int)k;

    for (size_t j = 0; remaining_k>0&&j<k; ++j) {
      NNCell const& cell = row[j];
      int cellcard = (int)cell.idx_label_v.size();
      if (cellcard<=remaining_k) {
        // Get them all
        remaining_k = remaining_k - cellcard;
        assert(remaining_k>=0);
        for (auto [idx, label] : cell.idx_label_v) {
          votes[label] += 1;
          votes_dist[label] += cell.distance;
        }
      } else {
        // Sample in the cell (thx c++17)
        std::vector<std::tuple<size_t, tempo::EL>> samples;
        std::sample(cell.idx_label_v.begin(), cell.idx_label_v.end(), std::back_inserter(samples), remaining_k, prng);
        remaining_k = 0;
        for (auto [idx, label] : samples) {
          votes[label] += 1;
          votes_dist[label] += cell.distance;
        }
      }
    }

    // Get classes with highest votes, split ties with smallest cumulative distance
    double bsf = tempo::utils::PINF;
    size_t maxvote = 0;
    std::vector<tempo::EL> best_classes;

    for (auto [label, nbv] : votes) {
      assert(votes_dist.contains(label));
      double dist = votes_dist[label];
      if (nbv>maxvote||(nbv==maxvote&&dist<bsf)) {
        maxvote = nbv;
        bsf = dist;
        best_classes.clear();
        best_classes.push_back(label);
      }
    }

    tempo::EL true_class = split.label(i).value();
    tempo::EL prediction = tempo::utils::pick_one(best_classes, prng);
    if (prediction==true_class) { nbcorrect++; }
  }

  return nbcorrect;

}

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
    if (cell.distance==dist) { cell.idx_label_v.emplace_back(idx, el); }
    else {
      // Replacement by 'insertion at head + removal of the last'
      row.insert(row.begin() + ipos, NNCell{std::vector<std::tuple<size_t, tempo::EL>>{{idx, el}}, dist});
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
    if (dist<=bsf) {
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
    if (dist<=bsf) {
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

  Json::Value jv;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Parse Arg and setup
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if (args.empty()) { do_exit(0, usage); }
  Config conf;

  // --- --- --- 1: Optional args
  cmd_optional(args, conf);

  // --- --- --- 2: Load dataset (must be done before 'check_transform')

  fs::path UCRPATH;
  string dsname;
  {
    auto parg_UCRPATH = tempo::scli::get_parameter<string>(args, "-p", tempo::scli::extract_string);
    if (!parg_UCRPATH) { do_exit(1, "Specify the UCR dataset -p flag, e.g. -p:/path/to/dataset:name"); }
    std::vector<std::string> v = tempo::reader::split(parg_UCRPATH.value(), ':');
    bool ok = v.size()==2;
    if (ok) {
      // Extract params
      UCRPATH = fs::path(v[0]);
      dsname = v[1];
      // Load the datasets
      fs::path dspath = UCRPATH/dsname;
      // Load train set
      auto train = reader::load_dataset_ts(dspath/(dsname + "_TRAIN.ts"), "train");
      if (train.index()==0) { do_exit(2, "Could not load the train set: " + std::get<0>(train)); }
      conf.loaded_train_split = std::get<1>(train);
      // Load test set
      auto test =
        reader::load_dataset_ts(dspath/(dsname + "_TEST.ts"), "test", conf.loaded_train_split.header().label_encoder());
      if (test.index()==0) { do_exit(2, "Could not load the test set: " + std::get<0>(test)); }
      conf.loaded_test_split = std::get<1>(test);
    }
    // Catchall
    if (!ok) { do_exit(1, "UCR Dataset (-p) parameter error"); }
  }

  auto const& train_header = conf.loaded_train_split.header();
  const auto train_top = train_header.size();

  auto const& test_header = conf.loaded_test_split.header();
  const auto test_top = test_header.size();

  // Sanity check
  {
    std::vector<std::string> errors = {};

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    auto [bcm, remainder] = conf.loaded_train_split.get_BCM();

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
      jv = conf.to_json();
      jv["status"] = "error";
      jv["status_message"] = utils::cat(errors, "; ");
      cout << jv.toStyledString() << endl;
      if (conf.outpath) {
        auto out = ofstream(conf.outpath.value());
        out << jv << endl;
      }
      exit(2);
    }
  }

  // --- --- --- 4: Check normalisation (must be done before transformation)
  cmd_normalisation(args, conf);

  // --- --- --- 4: Check transformation (must be done before check_dist)
  cmd_transform(args, conf);

  // --- --- --- 5: Check distance
  cmd_dist(args, conf);

  // Update info in json
  jv = conf.to_json();


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Computation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- --- --- ---
  // Progress reporting
  utils::ProgressMonitor pm(train_top + test_top);  // How many to do, both train and test accuracy
  size_t nbdone = 0;                                // How many done up to "now"

  // --- --- --- --- --- ---
  // Train table
  ResultTable train_table(train_top, conf.k);
  {
    // Multithreading control
    utils::ParTasks ptasks;
    std::mutex mutex;

    // Task implemented over the train exemplars
    auto task = [&](size_t train_idx) {
      std::vector<NNCell> results = nnk_train(conf.dist_fun, conf.train_split, train_idx, conf.k);
      {
        std::lock_guard lock(mutex);
        nbdone++;
        pm.print_progress(cout, nbdone);
        train_table.row(train_idx) = move(results);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    p.execute(conf.nbthreads, task, 0, train_top, 1);
  }

  // --- --- --- --- --- ---
  // Test table
  ResultTable test_table(test_top, conf.k);
  {
    // Multithreading control
    utils::ParTasks ptasks;
    std::mutex mutex;

    // Task implemented over the test exemplars
    auto task = [&](size_t test_idx) {
      auto const& self = conf.test_split[test_idx];
      std::vector<NNCell> results = nnk_test(conf.dist_fun, conf.train_split, self, conf.k);
      {
        std::lock_guard lock(mutex);
        nbdone++;
        pm.print_progress(cout, nbdone);
        test_table.row(test_idx) = move(results);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    p.execute(conf.nbthreads, task, 0, test_top, 1);
  }

  /*
  std::cout << std::endl << "TRAIN TABLE";
  for (size_t i = 0; i<train_table.table.size(); ++i) {
    std::cout << std::endl << i << " cl " << conf.train_split.label(i).value();

    for (const auto& cell : train_table.table[i]) {
      assert(cell.idxs.size()==cell.classes.size());
      std::cout << " | " << cell.distance << " ";
      for (auto [idx, label] : cell.idx_label_v) { std::cout << "(" << idx << ", " << label << ") "; }
    }
  }
  std::cout << std::endl << std::endl;


  std::cout << "TEST TABLE";
  for (size_t i = 0; i<test_table.table.size(); ++i) {
    std::cout << std::endl << i << " cl " << conf.test_split.label(i).value();

    for (const auto& cell : test_table.table[i]) {
      assert(cell.idxs.size()==cell.classes.size());
      std::cout << " | " << cell.distance << " ";
      for (auto [idx, label]: cell.idx_label_v) { std::cout << "(" << idx << ", " << label << ") "; }
    }
  }
  std::cout << std::endl;
   */

  // --- --- --- --- --- ---
  // JSON Tables
  auto get_train_label = [&](int i) -> tempo::EL { return conf.train_split.label(i).value(); };
  jv["train_table"] = train_table.to_json(get_train_label);

  auto get_test_label = [&](int i) -> tempo::EL { return conf.test_split.label(i).value(); };
  jv["test_table"] = test_table.to_json(get_test_label);


  //--- --- --- --- --- ---
  // Accuracy
  {
    // Train accuracy
    Json::Value train_accuracy = Json::objectValue;
    Json::Value train_01loss_nbc = Json::arrayValue;
    Json::Value train_01loss_acc = Json::arrayValue;
    for (size_t kk = 1; kk<=conf.k; ++kk) {
      size_t nbc = nb_correct_01loss(kk, train_table, conf.train_split, *conf.pprng);
      double acc = (double)(nbc)/(double)(train_top);
      train_01loss_nbc.append((int)nbc);
      train_01loss_acc.append(acc);
    }
    train_accuracy["nb_correct"] = train_01loss_nbc;
    train_accuracy["accuracy"] = train_01loss_acc;
    jv["01loss_train"] = train_accuracy;
  }

  {
    // Test accuracy
    Json::Value test_accuracy = Json::objectValue;
    Json::Value test_01loss_nbc = Json::arrayValue;
    Json::Value test_01loss_acc = Json::arrayValue;
    for (size_t kk = 1; kk<=conf.k; ++kk) {
      size_t nbc = nb_correct_01loss(kk, test_table, conf.test_split, *conf.pprng);
      double acc = (double)(nbc)/(double)(test_top);
      test_01loss_nbc.append((int)nbc);
      test_01loss_acc.append(acc);
    }
    test_accuracy["nb_correct"] = test_01loss_nbc;
    test_accuracy["accuracy"] = test_01loss_acc;
    jv["01loss_test"] = test_accuracy;
  }



  // --- --- --- --- --- ---
  // Output
  cout << endl << jv.toStyledString() << endl;
  if (conf.outpath) {
    auto out = ofstream(conf.outpath.value());
    out << jv << endl;
  }

  return 0;
}
