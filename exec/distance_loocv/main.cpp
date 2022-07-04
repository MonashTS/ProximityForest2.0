#include <string>
#include <vector>

#include <tempo/utils/utils.hpp>
#include <tempo/reader/new_reader.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/loocv/partable/partable.hpp>
#include <tempo/distance/helpers.hpp>
#include <tempo/distance/elastic/adtw.hpp>
#include <tempo/distance/lockstep/direct.hpp>

#include "cli.hpp"

int main(int argc, char **argv) {
  using namespace std;

  const size_t SAMPLE_SIZE = 4000;
  const double omega_exp = 5;

  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if (argc<6) {
    cout << "<path to ucr> <dataset name> <ge> <nbthreads> <output> required" << endl;
    exit(1);
  }

  // --- --- ---
  // Check arguments

  random_device rd;
  tempo::PRNG prng(rd());

  vector<string> argList(argv, argv + argc);
  filesystem::path path_ucr(argList[1]);
  string dataset_name(argList[2]);
  const double ge = stod(argList[3]);
  size_t nbthreads = stoi(argList[4]);
  filesystem::path outpath(argList[5]);

  // --- --- ---
  // Prepare result
  Json::Value jv;
  std::ofstream outfile(outpath);

  // --- --- ---
  // Load the datasets
  tempo::DTS loaded_train_split;
  tempo::DTS loaded_test_split;
  {
    filesystem::path path_train = path_ucr/dataset_name/(dataset_name + "_TRAIN.ts");
    filesystem::path path_test = path_ucr/dataset_name/(dataset_name + "_TEST.ts");
    auto train = tempo::reader::load_dataset_ts(path_train, "train");
    if (train.index()==0) { do_exit(2, "Could not load the train set: " + std::get<0>(train)); }
    loaded_train_split = std::move(std::get<1>(train));
    // Load test set
    auto test = tempo::reader::load_dataset_ts(path_test, "test", loaded_train_split.header().label_encoder());
    if (test.index()==0) { do_exit(2, "Could not load the test set: " + std::get<0>(test)); }
    loaded_test_split = std::move(std::get<1>(test));
  }
  tempo::DatasetHeader const& train_header = loaded_train_split.header();
  const size_t train_top = train_header.size();

  tempo::DatasetHeader const& test_header = loaded_test_split.header();
  const size_t test_top = test_header.size();

  jv["train"] = train_header.to_json();
  jv["test"] = test_header.to_json();

  // --- --- ---
  // Sanity check for the dataset
  {
    std::vector<std::string> errors = {};

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    auto [bcm, remainder] = loaded_train_split.get_BCM();

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
      jv["status_message"] = tempo::utils::cat(errors, "; ");
      cout << jv.toStyledString() << endl;
      outfile << jv << endl;
      exit(2);
    }
  }

  // --- --- ---
  // Normalisation/Transform (todo)
  tempo::DTS train_dts = loaded_train_split;
  tempo::DTS test_dts = loaded_test_split;

  // --- --- ---
  // Create the cost function
  const auto cf = tempo::distance::univariate::ade<tempo::TSeries>(ge);

  // --- --- ---
  // Prepare distance function for LOOCV/distance
  using distfun = tempo::classifier::loocv::partable::distance_fun_t;
  using ubfun = tempo::classifier::loocv::partable::upperbound_fun_t;

  distfun distance_fun;
  ubfun upperbound_fun;
  size_t nb_params;

  // --- --- ---
  // ADTW LOOCV Parameters generation through sampling
  // Generate parameters
  using PType = tuple<double, double>;    // r, omega
  vector<PType> params;
  double sampled_mean_dist;
  {

    {
      tempo::utils::StddevWelford welford;
      std::uniform_int_distribution<> distrib(0, (int)train_dts.size() - 1);
      for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
        const auto& q = train_dts[distrib(prng)];
        const auto& s = train_dts[distrib(prng)];
        const auto& cost = tempo::distance::directa(q, s, cf, tempo::utils::PINF);
        welford.update(cost);
      }
      sampled_mean_dist = welford.get_mean();
      // Json record
      Json::Value j(Json::objectValue);
      j["mean"] = welford.get_mean();
      j["stddev"] = welford.get_stddev_s();
      j["size"] = SAMPLE_SIZE;
      jv["sampling"] = j;
      cout << j.toStyledString() << endl;
    }

    // Sample omega
    for (size_t i = 0; i<100; i += 1) {
      const double r = std::pow((double)i/100.0, omega_exp);
      const double omega = r*sampled_mean_dist;
      params.emplace_back(std::tuple{r, omega});
    }

    // ADTW distance
    distance_fun = [&](size_t q, size_t c, size_t pidx, double bsf) {
      tempo::TSeries const& s1 = train_dts[q];
      tempo::TSeries const& s2 = train_dts[c];
      const auto d = tempo::distance::adtw(s1, s2, cf, std::get<1>(params[pidx]), bsf);
      return d;
    };

    // ADTW Upper Bound
    upperbound_fun = [&](size_t q, size_t c, [[maybe_unused]] size_t pidx) {
      tempo::TSeries const& s1 = train_dts[q];
      tempo::TSeries const& s2 = train_dts[c];
      return tempo::distance::directa(s1, s2, cf, tempo::utils::PINF);
    };

    nb_params = params.size();
  }

  cout << "LOOCV NB params = " << nb_params << endl;



  // --- --- ---
  // Exec LOOCV
  tempo::utils::duration_t loocv_time;
  auto start = tempo::utils::now();
  auto [loocv_params, loocv_nbcorrect] = tempo::classifier::loocv::partable::loocv(
    nb_params,
    train_header,
    distance_fun,
    upperbound_fun,
    nbthreads
  );
  loocv_time = tempo::utils::now() - start;

  // --- --- ---
  // Prepare test function
  using testfun = std::function<double(tempo::TSeries const& a, tempo::TSeries const& b, double bsf)>;
  testfun distance_test_fun;

  // --- --- ---
  // Prepare JSON output

  double train_accuracy = (double)loocv_nbcorrect/(double)train_top;
  Json::Value j_loocv;
  j_loocv["timing_ns"] = loocv_time.count();
  j_loocv["timing_human"] = tempo::utils::as_string(loocv_time);
  j_loocv["nb_correct"] = loocv_nbcorrect;
  j_loocv["accuracy"] = train_accuracy;


  // --- --- ---
  // ADTW result computation
  {

    // --- --- ---
    // Get result out of LOOCV
    double best_r;
    double best_omega;
    {
      // All best parameters have the same accuracy
      cout << dataset_name
           << " Best parameter(s): " << loocv_nbcorrect << "/" << train_top
           << " = " << train_accuracy << endl;
      // Report all the best in order - extract r and omega for json report
      std::sort(loocv_params.begin(), loocv_params.end());
      std::vector<double> all_r;
      std::vector<double> all_omega;
      for (const auto& pi : loocv_params) {
        auto [r, omega] = params[pi];
        all_r.push_back(r);
        all_omega.push_back(omega);
        cout << "  ratio = " << r << " penalty = " << omega << endl;
      }
      // Pick the median as "the best"
      {
        auto size = loocv_params.size();
        if (size%2==0) { // Median "without a middle" = average
          best_r = (get<0>(params[loocv_params[size/2 - 1]]) + get<0>(params[loocv_params[size/2]]))/2;
        } else { // Median "middle"
          best_r = get<0>(params[loocv_params[size/2]]);
        }
      }
      cout << dataset_name << " Pick median ratio g=" << best_r << endl;
      best_omega = best_r*sampled_mean_dist;
      // Record all parameters
      j_loocv["all_r"] = tempo::utils::to_json(all_r);
      j_loocv["all_omega"] = tempo::utils::to_json(all_omega);
      j_loocv["best_r"] = best_r;
      j_loocv["best_omega"] = best_omega;
      // Results
      jv["loocv_results"] = j_loocv;
    }

    distance_test_fun = [&](tempo::TSeries const& query, tempo::TSeries const& candidate, double bsf) {
      return tempo::distance::adtw(query, candidate, cf, best_omega, bsf);
    };

    {
      Json::Value j;
      j["name"] = "adtw";
      j["omega"] = best_omega;
      jv["distance"] = j;
    }

  }




  // --- --- ---
  // Test Accuracy
  size_t test_nb_correct = 0;
  tempo::utils::duration_t test_time{0};
  {
    // --- --- ---
    // Progress reporting
    tempo::utils::ProgressMonitor pm(test_top);    // How many to do, both train and test accuracy
    size_t nb_done = 0;                      // How many done up to "now"

    // --- --- ---
    // Accuracy variables
    // Multithreading control
    tempo::utils::ParTasks ptasks;
    std::mutex mutex;

    auto nn1_test_task = [&](size_t test_idx) mutable {
      // --- Test exemplar
      tempo::TSeries const& self = test_dts[test_idx];
      // --- 1NN bests with tie management
      double bsf = tempo::utils::PINF;
      std::set<tempo::EL> labels{};
      // --- 1NN Loop
      for (size_t train_idx{0}; train_idx<train_top; train_idx++) {
        tempo::TSeries const& candidate = train_dts[train_idx];
        double d = distance_test_fun(self, candidate, bsf);
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
        tempo::EL result = -1;
        std::sample(labels.begin(), labels.end(), &result, 1, prng);
        assert(result!=-1);
        if (result==test_header.label(test_idx).value()) { ++test_nb_correct; }
        nb_done++;
        pm.print_progress(std::cout, nb_done);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    auto test_start = tempo::utils::now();
    p.execute(nbthreads, nn1_test_task, 0, test_top, 1);
    test_time = tempo::utils::now() - test_start;
  }

  // --- Generate results
  double test_accuracy = ((double)test_nb_correct)/(double)test_top;
  cout << endl;
  cout << dataset_name << " NN1 test result: " << test_nb_correct << "/" << test_top << " = " << test_accuracy
       << "  (" << tempo::utils::as_string(test_time) << ")" << endl;

  // --- --- ---
  // --- --- --- Output
  // --- --- ---

  jv["status"] = "success";

  {
    Json::Value j;
    j["nb_correct"] = test_nb_correct;
    j["accuracy"] = test_accuracy;
    j["timing_ns"] = test_time.count();
    j["timing_human"] = tempo::utils::as_string(test_time);
    jv["test_results"] = j;
  }

  std::cout << dataset_name << " output to " << outpath << endl;

  cout << jv.toStyledString() << endl;
  outfile << jv << endl;

  return 0;
}
