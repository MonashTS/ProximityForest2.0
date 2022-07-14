#include "pch.h"
#include "tempo/reader/new_reader.hpp"

#include "tempo/classifier/SForest/stree.hpp"
//
#include <tempo/classifier/SForest/splitter/nn1/nn1splitters.hpp>
#include <tempo/classifier/SForest/splitter/nn1/MPGenerator.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_da.hpp>
#include "tempo/classifier/SForest/splitter/nn1/sp_adtw.hpp"
#include "tempo/classifier/SForest/splitter/nn1/sp_dtw.hpp"
#include "tempo/classifier/SForest/splitter/nn1/sp_wdtw.hpp"
#include "tempo/classifier/SForest/splitter/nn1/sp_erp.hpp"
#include "tempo/classifier/SForest/splitter/nn1/sp_lcss.hpp"
//
#include <tempo/classifier/SForest/leaf/pure_leaf.hpp>
//
#include <tempo/classifier/SForest/splitter/meta/chooser.hpp>
//
#include <tempo/distance/helpers.hpp>

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) { std::cerr << msg.value() << std::endl; }
  exit(code);
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;
  namespace SForest = tempo::classifier::SForest;
  namespace NN1Splitter = tempo::classifier::SForest::splitter::nn1;

  std::random_device rd;

  // ARGS
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);

  // Config
  fs::path UCRPATH(args[0]);
  string dataset_name(args[1]);
  size_t nbt = 100;
  size_t nbc = 5;
  size_t nbthread = 8;
  size_t train_seed = rd();
  size_t test_seed = train_seed + (nbt*nbc);

  // Json record
  Json::Value j;

  // Load UCR train and test dataset
  DTS train_dataset;
  DTS test_dataset;
  {
    auto start = utils::now();
    { // Read train
      fs::path train_path = UCRPATH/dataset_name/(dataset_name + "_TRAIN.ts");
      auto variant_train = tempo::reader::load_dataset_ts(train_path, "train");
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + train_path.string() + "': " + std::get<0>(variant_train)}); }
    }
    { // Read test
      fs::path test_path = UCRPATH/dataset_name/(dataset_name + "_TEST.ts");
      auto variant_test = tempo::reader::load_dataset_ts(test_path, "test");
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

  DatasetHeader const& train_header = train_dataset.header();
  const size_t train_size = train_header.size();

  DatasetHeader const& test_header = test_dataset.header();
  const size_t test_size = test_header.size();

  struct state {
    PRNG prng;

    SForest::splitter::nn1::NN1SplitterState distance_splitter_state;

    explicit state(size_t seed) : prng(seed) {}

    explicit state(PRNG&& prng) : prng(prng) {}

    state branch_fork(size_t /* branch_idx */) { return state(move(prng)); }

    void branch_merge(state&& s) { prng = move(s.prng); }

  };
  static_assert(SForest::MainState<state>);
  static_assert(SForest::splitter::nn1::HasNN1SplitterState<state>);

  struct data {

    map<string, DTS> trainset;
    map<string, DTS> testset;

    inline data(map<string, DTS> trainset, map<string, DTS> testset) :
      trainset(move(trainset)),
      testset(move(testset)) {}

    /// Concept TrainData: access to the training set
    inline DTS get_train_dataset(const std::string& tname) const { return trainset.at(tname); }

    /// Concept TrainData: access to the training header
    inline DatasetHeader const& get_train_header() const { return trainset.begin()->second.header(); }

    /// NN1TestData concepts requirement
    inline DTS get_test_dataset(const std::string& tname) const { return testset.at(tname); }

    /// Concept TestData: access to the training header
    inline DatasetHeader const& get_test_header() const { return testset.begin()->second.header(); }

  };
  static_assert(SForest::TrainData<data>);
  static_assert(SForest::TestData<data>);


  // --- --- --- Train/Test data

  map<string, DTS> train_map;
  train_map["default"] = train_dataset;

  map<string, DTS> test_map;
  test_map["default"] = test_dataset;

  data train_test_data(train_map, test_map);


  // --- --- --- --- --- --- Train one tree

  // --- --- --- Train state
  auto train_state = std::make_unique<state>(train_seed);

  // --- --- --- Train BCM
  auto [train_bcm, train_bcm_remains] = train_dataset.get_BCM();
  if (!train_bcm_remains.empty()) {
    cerr << "Error: train instances without label!" << endl;
    exit(5);
  }

  // --- --- --- Build the getters

  /// Pick transform
  vector<string> transforms{"default"};
  NN1Splitter::TransformGetter<state> transform_getter = [&](state& state) -> string {
    return utils::pick_one(transforms, state.prng);
  };

  /// Pick Exponent
  vector<double> exponents{2.0};
  NN1Splitter::ExponentGetter<state> exp_getter = [&](state& state) -> double {
    return utils::pick_one(exponents, state.prng);
  };

  /// Random window computation function [0, Lmax/4]
  NN1Splitter::WindowGetter<state, data> window_getter = [](state& state, data const& data) -> size_t {
    const size_t win_top = std::ceil((double)data.get_train_header().length_max()/4.0);
    return std::uniform_int_distribution<size_t>(0, win_top)(state.prng);
  };

  /// ERP Gap Value *AND* LCSS epsilon.
  /// Random fraction of the dataset standard deviation, within [stddev/5, stddev[
  NN1Splitter::StatGetter<state, data> frac_stddev =
    [](state& state, data const& data, ByClassMap const& bcm, string const& transform_name) -> double {
      const auto& train_dataset = data.get_train_dataset(transform_name);
      auto stddev_ = stddev(train_dataset, bcm.to_IndexSet());
      return std::uniform_real_distribution<F>(0.2*stddev_, stddev_)(state.prng);
    };

  // --- --- --- Build node generators

  /// Direct Alignment
  auto nn1da_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
    make_shared<NN1Splitter::DAGen<state, data>>(transform_getter, exp_getter)
  );

  /// DTW
  auto nn1dtw_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
    make_shared<NN1Splitter::DTWGen<state, data>>(transform_getter, exp_getter, window_getter)
  );

  /// DTWfull
  auto nn1dtwfull_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
    make_shared<NN1Splitter::DTWfullGen<state, data>>(transform_getter, exp_getter)
  );

  /// ADTW
  auto nn1adtw_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
    make_shared<NN1Splitter::ADTWGen<state, data>>(transform_getter, exp_getter)
  );

  /// WDTW
  auto nn1wdtw_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
    make_shared<NN1Splitter::WDTWGen<state, data>>(transform_getter, exp_getter)
  );

  /// ERP
  auto nn1erp_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
    make_shared<NN1Splitter::ERPGen<state, data>>(transform_getter, exp_getter, window_getter, frac_stddev)
  );

  /// LCSS
  auto nn1lcss_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
    make_shared<NN1Splitter::LCSSGen<state, data>>(transform_getter, exp_getter, window_getter, frac_stddev)
  );


  auto chooser_gen = make_shared<SForest::splitter::meta::SplitterChooserGen<state, data, state, data>>(
    vector<shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>>>{
      nn1da_gen,
      nn1dtw_gen,
      nn1dtwfull_gen,
      nn1adtw_gen,
      nn1wdtw_gen,
      nn1erp_gen,
      nn1lcss_gen
    },
    nbc
  );

  // --- --- --- Build leaf stopper

  auto leafgen_pure = make_shared<SForest::leaf::PureLeaf_Gen<state, data, state, data>>();

  // --- --- --- Train!

  SForest::STreeTrainer<state, data, state, data> tree_trainer(leafgen_pure, chooser_gen);
  auto [train_state1, trained_tree] = tree_trainer.train(std::move(train_state), train_test_data, train_bcm);


  // --- --- --- --- --- --- Classification one tree

  // --- --- --- Test state
  auto test_state = std::make_unique<state>(test_seed);

  // --- --- --- Test!

  size_t nb_correct = 0;
  for (size_t test_idx = 0; test_idx<test_size; ++test_idx) {
    auto [test_state1, result] = trained_tree->predict(std::move(test_state), train_test_data, test_idx);
    test_state = std::move(test_state1);  // Transmit state
    EL predicted_label = arma::index_max(result.probabilities);
    if (predicted_label==test_header.label(test_idx)) { nb_correct++; }
  }

  cout << "Nb correct = " << nb_correct << "/" << test_size << endl;
  cout << "Accuracy   = " << (double)nb_correct/(double)test_size << endl;




  /*

  // --- --- --- PF2018
  // --- --- train
  classifier::PF2018 pf2018(nbt, nbc);
  {
    DTSMap trainset;
    trainset.insert({"default", train_dataset});
    auto train_derives = transform::derive(train_dataset, 1);
    trainset.insert({"d1", std::move(train_derives[0])});
    // actual train call.
    const auto train_start = utils::now();
    pf2018.train(trainset, train_seed, nbthread);
    const auto train_delta = utils::now() - train_start;
    // Record JSON
  }

  // --- --- test
  {
    // Prepare data
    DTSMap testset;
    testset.insert({"default", test_dataset});
    auto test_derives = transform::derive(test_dataset, 1);
    testset.insert({"d1", std::move(test_derives[0])});
    // Prepare output
    arma::mat probabilities;
    arma::rowvec weights;
    // Call
    const auto test_start = utils::now();
    pf2018.predict(testset, probabilities, weights, test_seed, nbthread);
    const auto test_delta = utils::now() - test_start;
    // Record JSON
    const auto& test_header = test_dataset.header();
    const size_t test_top = test_header.size();
    Json::Value result_probas;
    Json::Value result_weights;
    Json::Value result_truelabels;
    // For each query (test instance)
    for (size_t query = 0; query<test_top; ++query) {
      // Store the probabilities
      result_probas.append(utils::to_json(probabilities.col(query)));
      // Store the weight associated with the probabilities
      result_weights.append(Json::Value(weights[query]));
      // Store the true label
      string true_l = test_header.labels()[query].value();
      size_t true_label_idx = test_header.label_encoder().label_to_index().at(true_l);
      result_truelabels.append(true_label_idx);
    }
    j["result_probabilities"] = result_probas;
    j["result_truelabels"] = result_truelabels;
    j["result_weights"] = result_weights;
  }

  // Output
  cout << j.toStyledString() << endl;

  return 0;
  */

}