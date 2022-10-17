#include <string>
#include <vector>

#include <tempo/utils/utils.hpp>
#include <tempo/utils/utils/stats.hpp>
#include <tempo/utils/readingtools.hpp>
#include <tempo/reader/reader.hpp>
#include <tempo/transform/tseries.univariate.hpp>
#include <tempo/distance/tseries.univariate.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/loocv/partable/partable.hpp>

#include <nlohmann/json.hpp>

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) { std::cerr << msg.value() << std::endl; }
  exit(code);
}

int main(int argc, char **argv) {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // CMDLine
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::string program_name(*argv);
  std::vector<std::string> args(argv + 1, argv + argc);
  if (argc<7) {
    std::cout << "<path to ucr> <dataset name> <transform> <distance:cfe> <nbthreads> <output> required" << std::endl;
    exit(1);
  }
  std::vector<std::string> argList(argv, argv + argc);
  std::filesystem::path path_ucr(argList[1]);
  std::string dataset_name(argList[2]);
  std::string transform(argList[3]);
  std::string dist_cfe(argList[4]);
  size_t nbthreads = stoi(argList[5]);
  std::filesystem::path outpath(argList[6]);

  // Check distance
  std::vector<std::string> vdist_ge = tempo::reader::split(dist_cfe, ':');
  std::string distance_name = vdist_ge[0];
  double cfe = 1;
  if (vdist_ge.size()>1) {
    auto mb_cfe = tempo::reader::as_double(vdist_ge[1]);
    if (mb_cfe) {
      cfe = mb_cfe.value();
    } else {
      std::cout << "<distance:cfe>  <cfe> must be a double" << std::endl;
      exit(1);
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Prepare result
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  nlohmann::json jv;
  std::ofstream outfile(outpath);

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Load the datasets
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  tempo::DTS raw_train;
  tempo::DTS raw_test;
  {
    auto start = tempo::utils::now();

    auto trainpath = path_ucr/dataset_name/(dataset_name + "_TRAIN.ts");
    auto testpath = path_ucr/dataset_name/(dataset_name + "_TEST.ts");

    // --- --- --- Read train
    auto variant_train = tempo::reader::load_udataset_ts(trainpath, "train");
    if (variant_train.index()==1) { raw_train = std::get<1>(variant_train); }
    else { do_exit(1, {"Could not read train set '" + trainpath.string() + "': " + std::get<0>(variant_train)}); }

    // --- --- --- Read test
    auto variant_test = tempo::reader::load_udataset_ts(testpath, "test");
    if (variant_test.index()==1) { raw_test = std::get<1>(variant_test); }
    else { do_exit(1, {"Could not read train set '" + testpath.string() + "': " + std::get<0>(variant_test)}); }

    auto delta = tempo::utils::now() - start;
    nlohmann::json dataset;
    dataset["train"] = raw_train.header().to_json();
    dataset["test"] = raw_test.header().to_json();
    dataset["load_time_ns"] = delta.count();
    dataset["load_time_str"] = tempo::utils::as_string(delta);
    jv["dataset"] = dataset;
  } //

  tempo::DatasetHeader const& train_header = raw_train.header();
  const size_t train_size = train_header.size();
  auto [train_bcm, train_bcm_remains] = raw_train.get_BCM();

  tempo::DatasetHeader const& test_header = raw_test.header();
  const size_t test_size = test_header.size();
  auto [test_bcm, test_bcm_remains] = raw_test.get_BCM();

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Sanity Check
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  {
    std::vector<std::string> errors = {};

    if (!train_bcm_remains.empty()) {
      errors.emplace_back("Could not take the By Class Map for all train exemplar (exemplar without label)");
    }

    if (train_header.variable_length()||train_header.has_missing_value()) {
      errors.emplace_back("Train set: variable length or missing data");
    }

    if (test_header.has_missing_value()||test_header.variable_length()) {
      errors.emplace_back("Test set: variable length or missing data");
    }

    if (!errors.empty()) {
      jv["status"] = "error";
      jv["status_message"] = tempo::utils::cat(errors, "; ");
      std::cout << to_string(jv) << std::endl;
      outfile << jv << std::endl;
    }
  } // End of Sanity Check

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Check the transform
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  tempo::DTS train = raw_train;
  tempo::DTS test = raw_test;

  auto derive = [](tempo::TSeries const& ts) { return tempo::transform::univariate::derive(ts); };

  if (transform=="derivative") {
    {
      auto train_d = raw_train.transform().map_shptr<tempo::TSeries>(derive, transform);
      train = tempo::DTS("train", train_d);
    }
    {
      auto test_d = raw_test.transform().map_shptr<tempo::TSeries>(derive, transform);
      test = tempo::DTS("test", test_d);
    }
  } else if (transform=="raw") {
    //
  } else {
    do_exit(1, "Wrong transform name");
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Prepare the distances/argument range (EE style)
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::random_device rd;
  tempo::PRNG prng(rd());

  using testfun = std::function<double(tempo::TSeries const& a, tempo::TSeries const& b, double bsf)>;
  testfun distance_test_fun;

  if (distance_name=="ADTW") {
    const size_t SAMPLE_SIZE = 4000;
    const double omega_exp = 5;

    // --- --- --- Sampling
    tempo::utils::StddevWelford welford;
    std::uniform_int_distribution<> distrib(0, (int)train_size - 1);
    for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
      const auto& q = train[distrib(prng)];
      const auto& s = train[distrib(prng)];
      const double cost = tempo::distance::univariate::directa(q, s, cfe, tempo::utils::PINF);
      welford.update(cost);
    }
    // State updated here through mutable reference
    const double sampled_mean_dist = welford.get_mean();

    // --- --- --- Generate parameters
    using PType = std::tuple<size_t, double>;    // i, omega
    auto params = std::make_shared<std::vector<PType>>();
    // 0-98 : sampling
    for (size_t i = 0; i<99; ++i) {
      const double r = std::pow((double)i/100.0, omega_exp);
      const double omega = r*sampled_mean_dist;
      params->emplace_back(std::tuple{i, omega});
    }
    // 99: PINF
    params->emplace_back(std::tuple{99, tempo::utils::PINF});


    // --- --- --- Build data for LOOCV
    tempo::classifier::nn1loocv::dist_ft distance;
    distance = [params, cfe, &train](size_t query_idx, size_t candidate_idx, size_t param_idx, double bsf) -> double {
      const double penalty = std::get<1>(params->at(param_idx));
      return tempo::distance::univariate::adtw(train[query_idx], train[candidate_idx], cfe, penalty, bsf);
    };

    tempo::classifier::nn1loocv::distUB_ft distanceUB;
    distanceUB = [cfe, &train](size_t query_idx, size_t candidate_idx, size_t param_idx) -> double {
      return tempo::distance::univariate::directa(train[query_idx], train[candidate_idx], cfe, tempo::utils::PINF);
    };

    size_t nbparams = params->size();

    // --- Exec LOOCV
    tempo::utils::duration_t loocv_time;
    auto start = tempo::utils::now();
    auto [loocv_params, loocv_nbcorrect] =
      tempo::classifier::nn1loocv::partable(distance, distanceUB, train_header, nbparams, nbthreads);
    loocv_time = tempo::utils::now() - start;

    double train_accuracy = (double)loocv_nbcorrect/(double)train_size;

    // --- --- ---
    // Prepare JSON output
    {
      nlohmann::json j_loocv;
      j_loocv["timing_ns"] = loocv_time.count();
      j_loocv["timing_human"] = tempo::utils::as_string(loocv_time);
      j_loocv["nb_correct"] = loocv_nbcorrect;
      j_loocv["accuracy"] = train_accuracy;
      jv["loocv_results"] = j_loocv;
    }


    // --- --- --- Get penalty
    {
      // All best parameters have the same accuracy
      std::cout << dataset_name
                << " Best parameter(s): " << loocv_nbcorrect << "/" << train_size
                << " = " << train_accuracy << std::endl;

      // Report all the best in order - extract r and omega for json report
      // Note: sort on tuples, where first component is size_t
      std::sort(loocv_params.begin(), loocv_params.end());
      std::vector<size_t> all_index;
      std::vector<double> all_omega;
      double median_penalty;
      for (const auto& pi : loocv_params) {
        auto [i, omega] = params->at(pi);
        all_index.push_back(i);
        all_omega.push_back(omega);
        std::cout << "parameter " << i << " penalty = " << omega << std::endl;
      }

      // Pick the median as "the best"
      {
        auto size = loocv_params.size();
        if (size%2==0) { // Median "without a middle" = average
          const double omega1 = std::get<1>(params->at(loocv_params[size/2 - 1]));
          const double omega2 = std::get<1>(params->at(loocv_params[size/2]));
          median_penalty = (omega1 + omega2)/2.0;
        } else { // Median "middle"
          median_penalty = std::get<1>(params->at(loocv_params[size/2]));
        }
      }

      std::cout << dataset_name << " Pick median penalty = " << median_penalty << std::endl;

      // Warning: must copy microcontext as we exit the current lexical block before using this function
      distance_test_fun = [=](tempo::TSeries const& query, tempo::TSeries const& candidate, double bsf) -> double {
        return tempo::distance::univariate::adtw(query, candidate, cfe, median_penalty, bsf);
      };

      {
        nlohmann::json j;
        j["name"] = "adtw";
        j["sample_size"] = SAMPLE_SIZE;
        j["sample_value"] = sampled_mean_dist;
        j["penalty_exponent"] = omega_exp;
        j["param_indexes"] = tempo::utils::to_json(all_index);
        j["penalties"] = tempo::utils::to_json(all_omega);
        j["median_penalty"] = median_penalty;
        j["cost_function_exponent"] = cfe;
        jv["distance"] = j;
      }

    } // End of get penalty

  } // END OF ADTW



  /*
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Do LOOCV
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  assert(distance.operator bool());
  assert(UBdistance.operator bool());
  assert(nbparams != -1);
   */


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Test Accuracy
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  size_t test_nb_correct = 0;
  tempo::utils::duration_t test_time{0};
  {
    // --- --- ---
    // Progress reporting
    tempo::utils::ProgressMonitor pm(test_size);    // How many to do, both train and test accuracy
    size_t nb_done = 0;                             // How many done up to "now"

    // --- --- ---
    // Accuracy variables
    // Multithreading control
    tempo::utils::ParTasks ptasks;
    std::mutex mutex;

    auto nn1_test_task = [&](size_t test_idx) mutable {
      // --- Test exemplar
      tempo::TSeries const& self = test[test_idx];
      // --- 1NN bests with tie management
      double bsf = tempo::utils::PINF;
      std::set<tempo::EL> labels{};
      // --- 1NN Loop
      for (size_t train_idx{0}; train_idx<train_size; train_idx++) {
        tempo::TSeries const& candidate = train[train_idx];
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
        assert(0<=result&&result<train_header.nb_classes());
        if (result==test_header.label(test_idx).value()) { ++test_nb_correct; }
        nb_done++;
        pm.print_progress(std::cout, nb_done);
      }
    };

    // Create the tasks per tree. Note that we clone the state.
    tempo::utils::ParTasks p;
    auto test_start = tempo::utils::now();
    p.execute(nbthreads, nn1_test_task, 0, test_size, 1);
    test_time = tempo::utils::now() - test_start;
  }

  // --- Generate results
  double test_accuracy = ((double)test_nb_correct)/(double)test_size;
  std::cout << std::endl;
  std::cout << dataset_name << " NN1 test result: " << test_nb_correct << "/" << test_size << " = " << test_accuracy
            << "  (" << tempo::utils::as_string(test_time) << ")" << std::endl;



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // --- Output
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  jv["status"] = "success";

  {
    nlohmann::json j;
    j["nb_correct"] = test_nb_correct;
    j["accuracy"] = test_accuracy;
    j["timing_ns"] = test_time.count();
    j["timing_human"] = tempo::utils::as_string(test_time);
    jv["test_results"] = j;
  }

  std::cout << dataset_name << " output to " << outpath << std::endl;

  std::cout << jv.dump(2) << std::endl;
  outfile << jv << std::endl;

  return 0;
}
