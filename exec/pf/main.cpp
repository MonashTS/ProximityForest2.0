#include "pch.h"
#include <tempo/reader/new_reader.hpp>

#include <tempo/classifier/SForest/stree.hpp>
#include <tempo/classifier/SForest/sforest.hpp>
//
#include <tempo/classifier/SForest/splitter/nn1/nn1splitters.hpp>
#include <tempo/classifier/SForest/splitter/nn1/MPGenerator.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_da.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_adtw.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_dtw.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_wdtw.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_erp.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_lcss.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_lorentzian.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_msm.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_sbd.hpp>
#include <tempo/classifier/SForest/splitter/nn1/sp_twe.hpp>
//
#include <tempo/classifier/SForest/leaf/pure_leaf.hpp>
//
#include <tempo/classifier/SForest/splitter/meta/chooser.hpp>
//
#include <tempo/distance/helpers.hpp>
//
#include <tempo/transform/derivative.hpp>

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
  string which_pf(args[2]);
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
    size_t seed;
    PRNG prng;

    SForest::splitter::nn1::NN1SplitterState distance_splitter_state;

    explicit state(size_t seed) : seed(seed), prng(seed) {}

    explicit state(PRNG&& prng) : prng(prng) {}

    state branch_fork(size_t /* branch_idx */) { return state(move(prng)); }

    void branch_merge(state&& s) { prng = move(s.prng); }

    state forest_fork(size_t branch_idx) { return state(seed + branch_idx); }

    void forest_merge(state&& /* s */ ) {}

  };
  static_assert(SForest::TreeState<state>);
  static_assert(SForest::ForestState<state>);
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

  auto train_derive_t = std::make_shared<DatasetTransform<TSeries>>(
    std::move(tempo::transform::derive(train_dataset.transform(), 1).back())
  );
  DTS train_derive("train", train_derive_t);

  map<string, DTS> train_map;
  train_map["default"] = train_dataset;
  train_map["derivative1"] = train_derive;

  auto test_derive_t = std::make_shared<DatasetTransform<TSeries>>(
    std::move(tempo::transform::derive(test_dataset.transform(), 1).back())
  );
  DTS test_derive("test", test_derive_t);

  map<string, DTS> test_map;
  test_map["default"] = test_dataset;
  test_map["derivative1"] = test_derive;

  data train_test_data(train_map, test_map);


  // --- --- --- --- --- --- Prepare Train state/data/splitters

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
  vector<string> transforms{"default", "derivative1"};
  NN1Splitter::TransformGetter<state> transform_getter = [&](state& state) -> string {
    return utils::pick_one(transforms, state.prng);
  };

  /// Pick default transform
  NN1Splitter::TransformGetter<state> transform_getter_default = [&](state& /* state */) -> string {
    return "default";
  };

  /// Pick derivative 1 transform
  NN1Splitter::TransformGetter<state> transform_getter_derivative1 = [&](state& /* state */) -> string {
    return "derivative1";
  };


  /// Pick Exponent
  vector<double> exponents{2.0};
  NN1Splitter::ExponentGetter<state> exp_getter = [&](state& state) -> double {
    return utils::pick_one(exponents, state.prng);
  };

  /// Pick Exponent always at 2.0
  NN1Splitter::ExponentGetter<state> exp_sqed = [&](state& /* state */) -> double { return 2.0; };

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

  /// MSM costs
  NN1Splitter::MSMGen<state, data>::CostGetter msm_cost = [](state& state) -> double {
    constexpr size_t N = 100;
    double costs[N]{
      0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375,
      0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125,
      0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352,
      0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784,
      0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88,
      4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28,
      9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8,
      60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100
    };
    return utils::pick_one(costs, N, state.prng);
  };

  /// TWE nu parameters
  NN1Splitter::TWEGen<state, data>::Getter twe_nu = [](state& state) -> double {
    constexpr size_t N = 10;
    double nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
    return utils::pick_one(nus, N, state.prng);
  };

  /// TWE lambda parameters
  NN1Splitter::TWEGen<state, data>::Getter twe_lambda = [](state& state) -> double {
    constexpr size_t N = 10;
    double lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                      0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
    return utils::pick_one(lambdas, N, state.prng);
  };

  // --- --- --- Build node generators
  std::shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>> node_splitter_gen;

  if (which_pf=="22") {

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
      make_shared<NN1Splitter::ERPGen<state, data>>(transform_getter, exp_sqed, window_getter, frac_stddev)
    );

    /// LCSS
    auto nn1lcss_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::LCSSGen<state, data>>(transform_getter, exp_sqed, window_getter, frac_stddev)
    );

    /// Lorentzian
    auto nn1lorentzian_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::LorentzianGen<state, data>>(transform_getter)
    );

    /// MSM
    auto nn1msm_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::MSMGen<state, data>>(transform_getter, msm_cost)
    );

    /// SBD
    auto nn1sbd_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::SBDGen<state, data>>(transform_getter)
    );

    /// TWE
    auto nn1twe_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::TWEGen<state, data>>(transform_getter, twe_nu, twe_lambda)
    );

    node_splitter_gen = make_shared<SForest::splitter::meta::SplitterChooserGen<state, data, state, data>>(
      vector<shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>>>{
        nn1da_gen,
        nn1dtw_gen,
        nn1dtwfull_gen,
        nn1adtw_gen,
        nn1wdtw_gen,
        nn1erp_gen,
        nn1lcss_gen,
        nn1lorentzian_gen,
        nn1msm_gen,
        nn1sbd_gen,
        nn1twe_gen
      },
      nbc
    );
  }
  else if (which_pf=="11") {

    /// Direct Alignment default
    auto nn1da_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::DAGen<state, data>>(transform_getter_default, exp_getter)
    );

    /// DTW default
    auto nn1dtw_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::DTWGen<state, data>>(transform_getter_default, exp_getter, window_getter)
    );

    /// DTW derivative1
    auto nn1dtw_d1_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::DTWGen<state, data>>(transform_getter_derivative1, exp_getter, window_getter)
    );

    /// DTWfull default
    auto nn1dtwfull_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::DTWfullGen<state, data>>(transform_getter_default, exp_getter)
    );

    /// DTWfull derivative1
    auto nn1dtwfull_d1_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::DTWfullGen<state, data>>(transform_getter_derivative1, exp_getter)
    );

    /// WDTW default
    auto nn1wdtw_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::WDTWGen<state, data>>(transform_getter_default, exp_getter)
    );

    /// WDTW derivative1
    auto nn1wdtw_d1_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::WDTWGen<state, data>>(transform_getter_derivative1, exp_getter)
    );

    /// ERP
    auto nn1erp_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::ERPGen<state, data>>(transform_getter_default, exp_sqed, window_getter, frac_stddev)
    );

    /// LCSS
    auto nn1lcss_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::LCSSGen<state, data>>(transform_getter_default, exp_sqed, window_getter, frac_stddev)
    );

    /// MSM
    auto nn1msm_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::MSMGen<state, data>>(transform_getter_default, msm_cost)
    );

    /// TWE
    auto nn1twe_def_gen = make_shared<NN1Splitter::NN1SplitterGen<state, data, state, data>>(
      make_shared<NN1Splitter::TWEGen<state, data>>(transform_getter_default, twe_nu, twe_lambda)
    );

    node_splitter_gen = make_shared<SForest::splitter::meta::SplitterChooserGen<state, data, state, data>>(
      vector<shared_ptr<SForest::NodeSplitterGen_i<state, data, state, data>>>{
        nn1da_def_gen,
        nn1dtw_def_gen,
        nn1dtw_d1_gen,
        nn1dtwfull_def_gen,
        nn1dtwfull_d1_gen,
        nn1wdtw_def_gen,
        nn1wdtw_d1_gen,
        nn1erp_def_gen,
        nn1lcss_def_gen,
        nn1msm_def_gen,
        nn1twe_def_gen
      },
      nbc
    );

  }

  // --- --- --- Build leaf stopper

  auto pleaf_gen = make_shared<SForest::leaf::PureLeaf_Gen<state, data, state, data>>();


  // --- --- --- --- --- --- Train

  auto tree_trainer = std::make_shared<SForest::STreeTrainer<state, data, state, data>>(pleaf_gen, node_splitter_gen);
  SForest::SForestTrainer<state, data, state, data> forest_trainer(tree_trainer, nbt);

  auto [train_state1, trained_forest] = forest_trainer.train(move(train_state), train_test_data, train_bcm, nbthread);

  // --- --- --- --- --- --- Test

  // --- --- --- Test state
  auto test_state = std::make_unique<state>(test_seed);

  // --- --- --- Test!

  size_t nb_correct = 0;
  for (size_t test_idx = 0; test_idx<test_size; ++test_idx) {
    auto [test_state1, result] = trained_forest->predict(std::move(test_state), train_test_data, test_idx, nbthread);
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