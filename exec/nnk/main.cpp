#include "pch.h"

#include <tempo/utils/utils.hpp>
#include <tempo/classifier/splitting_forest/proximity_forest/pf2018.hpp>
#include <tempo/distance/elastic/dtw.hpp>
#include <tempo/distance/lockstep/direct.hpp>
#include <tempo/transform/normalization.hpp>

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) { std::cerr << msg.value() << std::endl; }
  exit(code);
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;

  std::random_device rd;

  // ARGS
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);

  // Config
  fs::path UCRPATH(args[0]);
  string dataset_name(args[1]);
  std::string distname(args[2]);
  size_t nbthread = 8;
  size_t train_seed = rd();
  size_t k = 20;

  // PRNG
  PRNG prng(train_seed);

  // Json record
  Json::Value j;

  // Load UCR train and test dataset - Load test with the label encoder from train!
  DTS train_dataset;
  DTS test_dataset;
  {
    auto start = utils::now();
    { // Read train
      fs::path train_path = UCRPATH/dataset_name/(dataset_name + "_TRAIN.ts");
      auto variant_train = reader::load_dataset_ts(train_path);
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + train_path.string() + "': " + std::get<0>(variant_train)}); }
    }
    { // Read test with the label encoder from train
      fs::path test_path = UCRPATH/dataset_name/(dataset_name + "_TEST.ts");
      auto variant_test = reader::load_dataset_ts(test_path, train_dataset.header().label_encoder());
      if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
      else { do_exit(1, {"Could not read train set '" + test_path.string() + "': " + std::get<0>(variant_test)}); }
    }
    auto delta = utils::now() - start;
    Json::Value dataset;
    dataset["train"] = train_dataset.header().to_json();
    dataset["test"] = test_dataset.header().to_json();
    dataset["load_time_ns"] = delta.count();
    dataset["load_time_str"] = utils::as_string(delta);
    j["dataset"] = dataset;
  }


  // --- --- ---

  struct nn {
    size_t candidate_nn_idx;
    size_t candidate_nn_class;
    double distance;
  };

  const auto& train_header = train_dataset.header();
  const size_t train_top = train_header.size();
  const auto& test_header = test_dataset.header();
  const size_t test_top = test_header.size();

  vector<vector<nn>> kresults(test_top);

  utils::ParTasks ptasks;
  utils::ProgressMonitor pm(test_top);


  // Multithreading control
  std::mutex mutex;
  size_t nbdone = 0;

  auto test_task = [&](size_t qidx) {

    TSeries const& query = test_dataset[qidx];
    double bsf = tempo::utils::PINF<double>; // bsf = worst of the knn
    std::vector<nn> results;
    results.reserve(k);
    // Take candidates in random order
    std::vector<size_t> candidate_idxs(train_top);
    std::iota(candidate_idxs.begin(), candidate_idxs.end(), 0);
    std::shuffle(candidate_idxs.begin(), candidate_idxs.end(), std::mt19937{std::random_device{}()});
    // Candidate loop
    for (size_t candidateidx : candidate_idxs) {
      TSeries const& candidate = train_dataset[candidateidx];
      double dist;
      // Test with square euclidean
      if (distname=="eucl") {
        dist = distance::directa(
          query.size(), candidate.size(),
          distance::univariate::ad2<F, TSeries>(query, candidate),
          bsf
        );
      } else if (distname=="dtw") {
        dist = distance::dtw(
          query.size(), candidate.size(),
          distance::univariate::ad2<F, TSeries>(query, candidate),
          utils::NO_WINDOW,
          bsf
        );
      } else {
        utils::should_not_happen();
      }
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
  for (size_t i = 0; i<test_top; ++i) {
    p.push_task(test_task, i);
  }

  p.execute(nbthread);

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

  j["result"] = jtest;

  cout << j.toStyledString() << endl;

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

      if (selected_class==true_class) {
        nbcorrect++;
      }

    }

    cout << "k = " << kk << " nb correct = " << nbcorrect << "/" << test_top
      << " = " << (double)nbcorrect/double(test_top) << endl;

  }

  return 0;
}
