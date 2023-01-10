#include <tempo/reader/reader.hpp>
#include <nlohmann/json.hpp>

#include "cli.hpp"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;
  using namespace tempo::utils;

  nlohmann::json jv;

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
      auto train = tempo::reader::load_udataset_ts(dspath/(dsname + "_TRAIN.ts"), "train");
      if (train.index()==0) { do_exit(2, "Could not load the train set: " + std::get<0>(train)); }
      conf.loaded_train_split = std::move(std::get<1>(train));
      // Load test set
      auto test = tempo::reader::load_udataset_ts(dspath/(dsname + "_TEST.ts"), "test",
                                                 conf.loaded_train_split.header().label_encoder());
      if (test.index()==0) { do_exit(2, "Could not load the test set: " + std::get<0>(test)); }
      conf.loaded_test_split = std::move(std::get<1>(test));
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
      cout << to_string(jv) << endl;
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
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  if (conf.opair) {
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Computation Pairwise
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    auto const [src1, idx1, src2, idx2] = conf.opair.value();
    const TSeries& t1 = (src1=="train") ? conf.train_split[idx1] : conf.test_split[idx1];
    const TSeries& t2 = (src2=="train") ? conf.train_split[idx2] : conf.test_split[idx2];
    double d = conf.dist_fun(t1, t2, utils::PINF);

    // --- --- --- Test Accuracy
    {
      nlohmann::json j;
      j["t1"] = src1+":"+std::to_string(idx1);
      j["t2"] = src2+":"+std::to_string(idx2);
      j["value"] = d;
      jv["result"] = j;
    }

    // --- --- --- Output
    jv["status"] = "success";

  } else if(conf.do_train_acc){
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Computation NN1 train accuracy
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- --- --- ---
    // Train accuracy
    size_t train_nb_correct = 0;
    auto now = utils::now();

    // --- --- ---
    // Progress reporting
    utils::ProgressMonitor pm(train_top);    // How many to do
    size_t nb_done = 0;                      // How many done up to "now"

    // --- --- ---
    // Accuracy variables
    // Multithreading control
    utils::ParTasks ptasks;
    std::mutex mutex;

    auto nn1_train_task = [&](size_t train_idx) mutable {
      // --- Test exemplar
      TSeries const& self = conf.train_split[train_idx];
      // --- 1NN bests with tie management
      double bsf = tempo::utils::PINF;
      std::set<tempo::EL> labels{};
      // --- 1NN Loop
      for (size_t other_train_idx{0}; other_train_idx<train_top; other_train_idx++) {
        if(other_train_idx == train_idx){continue;}
        TSeries const& candidate = conf.train_split[other_train_idx];
        double d = conf.dist_fun(self, candidate, bsf);
        if (d<bsf) { // Best: clear labels and insert new
          labels.clear();
          labels.insert(train_header.label(other_train_idx).value());
          bsf = d;
        } else if (d==bsf) { // Same: add label in
          labels.insert(train_header.label(other_train_idx).value());
        }
      }
      // --- Update accuracy
      {
        std::lock_guard lock(mutex);
        tempo::EL result = std::numeric_limits<EL>::max();
        std::sample(labels.begin(), labels.end(), &result, 1, *conf.pprng);
        assert(result!=std::numeric_limits<EL>::max());
        if (result==train_header.label(train_idx).value()) { ++train_nb_correct; }
        nb_done++;
        pm.print_progress(std::cout, nb_done);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    p.execute(conf.nbthreads, nn1_train_task, 0, train_top, 1);

    auto elapsed = utils::now() - now;

    // --- --- --- --- --- ---
    // Output

    // --- --- --- Train Accuracy
    {
      nlohmann::json j;
      j["nb_correct"] = train_nb_correct;
      j["accuracy"] = (double)train_nb_correct/(double)(train_top);
      j["time_ns"] = elapsed.count();
      j["time_human"] = utils::as_string(elapsed);
      jv["01_loss_train"] = j;
    }

    // --- --- --- Output
    jv["status"] = "success";


  }
  else {
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Computation NN1 test accuracy
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- --- --- ---
    // Test accuracy
    size_t test_nb_correct = 0;
    auto now = utils::now();

    // --- --- ---
    // Progress reporting
    utils::ProgressMonitor pm(test_top);    // How many to do, both train and test accuracy
    size_t nb_done = 0;                      // How many done up to "now"

    // --- --- ---
    // Accuracy variables
    // Multithreading control
    utils::ParTasks ptasks;
    std::mutex mutex;

    auto nn1_test_task = [&](size_t test_idx) mutable {
      // --- Test exemplar
      TSeries const& self = conf.test_split[test_idx];
      // --- 1NN bests with tie management
      double bsf = tempo::utils::PINF;
      std::set<tempo::EL> labels{};
      // --- 1NN Loop
      for (size_t train_idx{0}; train_idx<train_top; train_idx++) {
        TSeries const& candidate = conf.train_split[train_idx];
        double d = conf.dist_fun(self, candidate, bsf);
        if (d<bsf) { // Best: clear labels and insert new
          labels.clear();
          labels.insert(train_header.label(train_idx).value());
          bsf = d;
        } else if (d==bsf) { // Same: add label in
          labels.insert(train_header.label(train_idx).value());
        }
      }
      // --- Update accuracy
      {
        std::lock_guard lock(mutex);
        tempo::EL result = std::numeric_limits<EL>::max();
        std::sample(labels.begin(), labels.end(), &result, 1, *conf.pprng);
        assert(result!=std::numeric_limits<EL>::max());
        if (result==test_header.label(test_idx).value()) { ++test_nb_correct; }
        nb_done++;
        pm.print_progress(std::cout, nb_done);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    p.execute(conf.nbthreads, nn1_test_task, 0, test_top, 1);

    auto elapsed = utils::now() - now;

    // --- --- --- --- --- ---
    // Output

    // --- --- --- Test Accuracy
    {
      nlohmann::json j;
      j["nb_correct"] = test_nb_correct;
      j["accuracy"] = (double)test_nb_correct/(double)(test_top);
      j["time_ns"] = elapsed.count();
      j["time_human"] = utils::as_string(elapsed);
      jv["01_loss_test"] = j;
    }

    // --- --- --- Output
    jv["status"] = "success";
  }

  cout << endl << jv.dump(2) << endl;
  if (conf.outpath) {
    auto out = ofstream(conf.outpath.value());
    out << jv.dump(2) << endl;
  }

  return 0;
}
