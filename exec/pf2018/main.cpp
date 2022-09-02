#include <tempo/utils/readingtools.hpp>
#include <tempo/reader/new_reader.hpp>
#include <tempo/transform/univariate.hpp>

#include "tempo/classifier/TSChief/tree.hpp"
#include "tempo/classifier/TSChief/forest.hpp"
#include "tempo/classifier/TSChief/sleaf/pure_leaf.hpp"
#include "tempo/classifier/TSChief/snode/meta/chooser.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1splitter.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_directa.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_adtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_dtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_dtwfull.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_wdtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_erp.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_lcss.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_msm.hpp"

#include "cmdline.hpp"

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) { std::cerr << msg.value() << std::endl; }
  exit(code);
}

int main(int argc, char **argv) {
  // --- --- --- Type / namespace
  using namespace std;
  using namespace tempo;

  // --- --- --- Randomness
  std::random_device rd;
  size_t train_seed = rd();
  size_t test_seed = rd();
  size_t tiebreak_seed = rd();

  // --- --- --- Prepare JSon record for output
  Json::Value jv;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Read args
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  cmdopt opt;
  {
    variant<string, cmdopt> mb_opt = parse_cmd(argc, argv);
    switch (mb_opt.index()) {
    case 0: {
      cerr << "Error: " << std::get<0>(mb_opt) << std::endl;
      exit(1);
    }
    case 1: { opt = std::get<1>(mb_opt); }
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Read dataset
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- Load UCR train and test dataset
  DTS train_dataset;
  DTS test_dataset;
  {
    auto start = utils::now();

    if (opt.input.index()==0) {
      read_ucr ru = std::get<0>(opt.input);
      // --- --- --- Read train
      auto variant_train = tempo::reader::load_dataset_ts(ru.train, "train");
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + ru.train.string() + "': " + std::get<0>(variant_train)}); }
      // --- --- --- Read test
      auto variant_test = tempo::reader::load_dataset_ts(ru.test, "test");
      if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
      else { do_exit(1, {"Could not read train set '" + ru.test.string() + "': " + std::get<0>(variant_test)}); }
    } else if (opt.input.index()==1) {
      read_csv rc = std::get<1>(opt.input);
      // --- --- --- Read train
      auto variant_train = tempo::reader::load_dataset_csv(rc.train, rc.name, 1, "train", {}, rc.skip_header, rc.sep);
      if (variant_train.index()==1) { train_dataset = std::get<1>(variant_train); }
      else { do_exit(1, {"Could not read train set '" + rc.train.string() + "': " + std::get<0>(variant_train)}); }
      // --- --- --- Read test
      auto variant_test = tempo::reader::load_dataset_csv(rc.test, rc.name, 1, "test", {}, rc.skip_header, rc.sep);
      if (variant_test.index()==1) { test_dataset = std::get<1>(variant_test); }
      else { do_exit(1, {"Could not read test set '" + rc.test.string() + "': " + std::get<0>(variant_test)}); }
    } else { tempo::utils::should_not_happen(); }

    auto delta = utils::now() - start;
    Json::Value dataset;
    dataset["train"] = train_dataset.header().to_json();
    dataset["test"] = test_dataset.header().to_json();
    dataset["load_time_ns"] = delta.count();
    dataset["load_time_str"] = utils::as_string(delta);
    jv["dataset"] = dataset;
  } // End of dataset loading

  DatasetHeader const& train_header = train_dataset.header();
  DatasetHeader const& test_header = test_dataset.header();

  auto [train_bcm, train_bcm_remains] = train_dataset.get_BCM();

  // --- --- --- Sanity check
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
      jv["status_message"] = utils::cat(errors, "; ");
      cout << jv.toStyledString() << endl;
      if (opt.output) {
        auto out = ofstream(opt.output.value());
        out << jv << endl;
      }
      exit(1);
    }

  } // End of Sanity Check


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Prepare data and state for the PF configuration
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // TODO: To link with transforms, snode requirement, etc...
  // For now, just pre-compute everything, regardless of the actual needs

  namespace tsc = tempo::classifier::TSChief;
  namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

  // --- --- ---
  // --- --- --- Constants & configurations
  // --- --- ---

  // --- Time series transforms
  const std::string tr_default("default");
  const std::string tr_d1("derivative1");
  const std::string tr_d2("derivative2");

  // Make map of transforms (transform name, dataset)
  using MDTS = map<string, DTS>;
  shared_ptr<MDTS> train_map = make_shared<MDTS>();
  shared_ptr<MDTS> test_map = make_shared<MDTS>();

  auto prepare_data_start_time = utils::now();
  {
    namespace ttu = tempo::transform::univariate;

    // --- TRAIN
    auto train_derive_t1 = train_dataset.transform().map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), tr_d1);
    auto train_derive_t2 = train_derive_t1->map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), tr_d2);
    DTS train_derive_1("train", train_derive_t1);
    DTS train_derive_2("train", train_derive_t2);
    train_map->emplace("default", train_dataset);
    train_map->emplace(tr_d1, train_derive_1);
    train_map->emplace(tr_d2, train_derive_2);

    // --- TEST
    auto test_derive_t1 = test_dataset.transform().map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), tr_d1);
    auto test_derive_t2 = test_derive_t1->map_shptr<TSeries>(TSeries::mapfun(ttu::derive<F>), tr_d2);
    DTS test_derive_1("test", test_derive_t1);
    DTS test_derive_2("test", test_derive_t2);
    test_map->emplace("default", test_dataset);
    test_map->emplace(tr_d1, test_derive_1);
    test_map->emplace(tr_d2, test_derive_2);
  }
  auto prepare_data_elapsed = utils::now() - prepare_data_start_time;

  // --- --- ---
  // --- --- --- DATA
  // --- --- ---

  tsc::TreeData tdata;

  // --- TRAIN
  std::shared_ptr<tsc::i_GetData<DatasetHeader>> get_train_header =
    tdata.register_data<DatasetHeader>(train_dataset.header_ptr());

  std::shared_ptr<tsc::i_GetData<std::map<std::string, DTS>>> get_train_data =
    tdata.register_data<map<string, DTS>>(train_map);

  // --- TEST
  std::shared_ptr<tsc::i_GetData<std::map<std::string, DTS>>> get_test_data =
    tdata.register_data<map<string, DTS>>(test_map);


  // --- --- ---
  // --- --- --- STATE
  // --- --- ---

  std::cout << "Train seed = " << train_seed << std::endl;
  tsc::TreeState tstate(train_seed, 0);

  // State for 1NN distance splitters - cache the indexset
  using GS1NNState = tsc_nn1::GenSplitterNN1_State;
  std::shared_ptr<tsc::i_GetState<GS1NNState>> get_GenSplitterNN1_State =
    tstate.register_state<GS1NNState>(std::make_unique<GS1NNState>());

  // ADTW train - cache sampling
  std::shared_ptr<tsc::i_GetState<tsc_nn1::ADTWGenState>> get_adtw_state =
    tstate.register_state<tsc_nn1::ADTWGenState>(std::make_unique<tsc_nn1::ADTWGenState>());


  // --- --- ---
  // --- --- --- Distance snode parameter space
  // --- --- ---

  // --- --- Exponent getters
  //const std::vector<F> dist_cfe_set{0.5, 1.0/1.5, 1, 1.5, 2};
  const std::vector<F> dist_cfe_set{0.5, 1, 2};
  tsc_nn1::ExponentGetter getter_cfe_set = [&](tsc::TreeState& s) { return utils::pick_one(dist_cfe_set, s.prng); };
  tsc_nn1::ExponentGetter getter_cfe_1 = [](tsc::TreeState& /* state */) { return 1.0; };
  tsc_nn1::ExponentGetter getter_cfe_2 = [](tsc::TreeState& /* state */) { return 2.0; };

  // --- --- Transform getters
  const std::vector<std::string> dist_tr_set{tr_default, tr_d1, tr_d2};
  tsc_nn1::TransformGetter getter_tr_set = [&](tsc::TreeState& s) { return utils::pick_one(dist_tr_set, s.prng); };
  tsc_nn1::TransformGetter getter_tr_default = [&](tsc::TreeState& /* state */) { return tr_default; };
  tsc_nn1::TransformGetter getter_tr_d1 = [&](tsc::TreeState& /* state */) { return tr_d1; };
  tsc_nn1::TransformGetter getter_tr_d2 = [&](tsc::TreeState& /* state */) { return tr_d2; };

  // --- --- Window getter
  tsc_nn1::WindowGetter getter_window = [&](tsc::TreeState& s, tsc::TreeData const& /* d */) {
    size_t maxl = train_header.length_max();
    const size_t win_top = std::floor((double)maxl + 1/4.0);
    return std::uniform_int_distribution<size_t>(0, win_top)(s.prng);
  };

  // --- --- ERP Gap Value *AND* LCSS epsilon.
  // Random fraction of the incoming data standard deviation, within [stddev/5, stddev[
  tsc_nn1::StatGetter frac_stddev =
    [&](tsc::TreeState& state, tsc::TreeData const& data, ByClassMap const& bcm, string const& transform_name) {
      const DTS& train_dataset = get_train_data->at(data).at(transform_name);
      auto stddev_ = stddev(train_dataset, bcm.to_IndexSet());
      return std::uniform_real_distribution<F>(stddev_/5.0, stddev_)(state.prng);
    };

  // --- --- --- MSM Cost
  tsc_nn1::T_GetterState<F> getter_msm_cost = [](tsc::TreeState& state) {
    constexpr size_t MSM_N = 100;
    constexpr F msm_cost[MSM_N]{
      0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375,
      0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125,
      0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352,
      0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784,
      0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88,
      4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28,
      9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8,
      60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100
    };
    return utils::pick_one(msm_cost, MSM_N, state.prng);
  };


  // --- --- --- TWE nu & lambda parameters
  tsc_nn1::T_GetterState<F> getter_twe_nu = [](tsc::TreeState& state) {
    constexpr size_t N = 10;
    constexpr F nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
    return utils::pick_one(nus, N, state.prng);
  };

  tsc_nn1::T_GetterState<F> getter_twe_lambda = [](tsc::TreeState& state) {
    constexpr size_t N = 10;
    constexpr F lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                           0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
    return utils::pick_one(lambdas, N, state.prng);
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Build the snode generators
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- Leaf Generator
  auto leaf_gen = std::make_shared<tsc::sleaf::GenLeaf_Pure>(get_train_header);

  // --- --- --- Node Generator
  vector<shared_ptr<tsc::i_GenNode>> generators;

  // --- NN1 distance node generator
  {
    // List of distance generators (this is specific to our collection of NN1 Splitter generators)
    vector<shared_ptr<tsc_nn1::i_GenDist>> gendist;

    // Direct Alignment
    gendist.push_back(make_shared<tsc_nn1::DAGen>(getter_tr_set, getter_cfe_set));

    // ADTW, with state and access to train data for sampling
    gendist.push_back(make_shared<tsc_nn1::ADTWGen>(getter_tr_set, getter_cfe_set, get_adtw_state, get_train_data));

    // DTW
    gendist.push_back(make_shared<tsc_nn1::DTWGen>(getter_tr_set, getter_cfe_set, getter_window));

    // WDTW
    gendist.push_back(make_shared<tsc_nn1::WDTWGen>(getter_tr_set, getter_cfe_set, train_header.length_max()));

    // DTWFull
    gendist.push_back(make_shared<tsc_nn1::DTWFullGen>(getter_tr_set, getter_cfe_set));

    // ERP
    gendist.push_back(make_shared<tsc_nn1::ERPGen>(getter_tr_set, getter_cfe_2, frac_stddev, getter_window));

    // LCSS
    gendist.push_back(make_shared<tsc_nn1::LCSSGen>(getter_tr_set, frac_stddev, getter_window));

    // MSM
    gendist.push_back(make_shared<tsc_nn1::MSMGen>(getter_tr_set, getter_msm_cost));

    // TWE
    //gendist.push_back(make_shared<tsc_nn1::TWEGen>(getter_tr_set, getter_twe_nu, getter_twe_lambda));

    // Wrap each distance generator in GenSplitter1NN (which is a i_GenNode) and push in generators
    for (auto const& gd : gendist) {
      generators.push_back(
        make_shared<tsc_nn1::GenSplitterNN1>(gd, get_GenSplitterNN1_State, get_train_data, get_test_data)
      );
    }
  }

  // --- Put a node chooser over all generators
  auto node_gen = make_shared<tsc::snode::meta::SplitterChooserGen>(std::move(generators), opt.nb_candidates);



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Make PF and use it
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- ---
  // --- --- --- Build the forest TODO: make the forest, one tree for now
  // --- --- ---

  auto tree_trainer = std::make_shared<tsc::TreeTrainer>(leaf_gen, node_gen);
  tsc::ForestTrainer forest_trainer(get_train_header, tree_trainer, opt.nb_trees);

  // --- --- ---
  // --- --- --- TRAIN
  // --- --- ---

  auto train_start_time = utils::now();
  auto forest = forest_trainer.train(tstate, tdata, train_bcm, opt.nb_threads, &std::cout);
  auto train_elapsed = utils::now() - train_start_time;

  // --- --- ---
  // --- --- --- TEST
  // --- --- ---

  classifier::ResultN result;
  auto test_start_time = utils::now();
  {
    const size_t test_size = test_dataset.size();
    for (size_t test_idx = 0; test_idx<test_size; ++test_idx) {
      // Get the prediction per tree
      std::vector<classifier::Result1> vecr = forest->predict(tstate, tdata, test_idx, opt.nb_threads);
      // Merge prediction as we want. Here, arithmetic average weighted by number of leafs
      // Result1 must be initialised with the number of classes!
      classifier::Result1 r1(train_header.nb_classes());
      for (const auto& r : vecr) {
        r1.probabilities += r.probabilities*r.weight;
        r1.weight += r.weight;
      }
      r1.probabilities /= r1.weight;
      //
      result.append(r1);
    }
  }
  auto test_elapsed = utils::now() - test_start_time;

  PRNG prng(tiebreak_seed);
  size_t nb_correct = result.nb_correct_01loss(test_header, IndexSet(test_header.size()), prng);
  double accuracy = (double)nb_correct/(double)test_header.size();


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Generate output and exit
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


  jv["status"] = "success";

  { // Classifier information
    Json::Value j;
    j["train_time_ns"] = train_elapsed.count();
    j["train_time_human"] = utils::as_string(train_elapsed);
    j["test_time_ns"] = test_elapsed.count();
    j["test_time_human"] = utils::as_string(test_elapsed);
    j["prepare_data_ns"] = prepare_data_elapsed.count();
    j["prepare_data_human"] = utils::as_string(prepare_data_elapsed);
    //
    jv["classifier"] = opt.pfconfig;
    jv["classifier_info"] = j;
  }

  { // 01 loss results
    Json::Value j;
    j["nb_corrects"] = nb_correct;
    j["accuracy"] = accuracy;
    jv["01loss"] = j;
  }

  cout << jv << endl;

  if (opt.output) {
    auto out = ofstream(opt.output.value());
    out << jv << endl;
  }

  return 0;

}