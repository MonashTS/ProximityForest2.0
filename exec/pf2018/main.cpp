#include <tempo/utils/readingtools.hpp>
#include <tempo/reader/new_reader.hpp>
#include <tempo/transform/univariate.hpp>

#include <tempo/classifier/sfdyn/stree.hpp>
#include <tempo/classifier/sfdyn/splitter/leaf/pure_leaf.hpp>
#include <tempo/classifier/sfdyn/splitter/node/meta/chooser.hpp>
#include <tempo/classifier/sfdyn/splitter/node/nn1/nn1_directa.hpp>
#include <tempo/classifier/sfdyn/splitter/node/nn1/nn1_dtwfull.hpp>
#include <tempo/classifier/sfdyn/splitter/node/nn1/MPGenerator.hpp>

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
  // TODO: To link with transforms, splitter requirement, etc...
  // For now, just pre-compute everything, regardless of the actual needs

  namespace tsf = tempo::classifier::sf;
  namespace tsf1nn = tempo::classifier::sf::node::nn1dist;

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

  // --- Main Distance splitter parameter space

  // --- --- Exponent getters
  const std::vector<F> dist_cfe_set{0.5, 1.0/1.5, 1, 1.5, 2};
  tsf1nn::ExponentGetter getter_cfe_set = [&](tsf::TreeState& s) { return utils::pick_one(dist_cfe_set, s.prng); };
  tsf1nn::ExponentGetter getter_cfe_1 = [](tsf::TreeState& /* state */) { return 1.0; };
  tsf1nn::ExponentGetter getter_cfe_2 = [](tsf::TreeState& /* state */) { return 2.0; };

  // --- --- Transform getters
  const std::vector<std::string> dist_tr_set{tr_default, tr_d1, tr_d2};
  tsf1nn::TransformGetter getter_tr_set = [&](tsf::TreeState& s) { return utils::pick_one(dist_tr_set, s.prng); };
  tsf1nn::TransformGetter getter_tr_default = [&](tsf::TreeState& /* state */) { return tr_default; };
  tsf1nn::TransformGetter getter_tr_d1 = [&](tsf::TreeState& /* state */) { return tr_d1; };
  tsf1nn::TransformGetter getter_tr_d2 = [&](tsf::TreeState& /* state */) { return tr_d2; };


  // --- --- ---
  // --- --- --- DATA
  // --- --- ---

  tsf::TreeData tdata;

  // --- TRAIN
  std::shared_ptr<tsf::i_GetData<DatasetHeader>> get_train_header =
    tdata.register_data<DatasetHeader>(train_dataset.header_ptr());

  std::shared_ptr<tsf::i_GetData<std::map<std::string, DTS>>> get_train_data =
    tdata.register_data<map<string, DTS>>(train_map);

  // --- TEST
  std::shared_ptr<tsf::i_GetData<std::map<std::string, DTS>>> get_test_data =
    tdata.register_data<map<string, DTS>>(test_map);


  // --- --- ---
  // --- --- --- STATE
  // --- --- ---

  tsf::TreeState tstate(train_seed, 0);

  // State for 1NN distance splitters - cache the indexset
  using GS1NNState = tsf1nn::GenSplitterNN1_State;
  std::shared_ptr<tsf::i_GetState<GS1NNState>> get_GenSplitterNN1_State =
    tstate.register_state<GS1NNState>(std::make_unique<GS1NNState>());


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Build the splitter generators
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- Leaf Generator
  auto leaf_gen = std::make_shared<tsf::leaf::GenLeaf_Pure>(get_train_header);

  // --- --- --- Node Generator
  vector<shared_ptr<tsf::i_GenNode>> generators;

  // --- NN1 distance node generator
  {
    // List of distance generators (this is specific to our collection of NN1 Splitter generators)
    vector<shared_ptr<tsf1nn::i_GenDist>> gendist;
    // Direct Alignment
    gendist.push_back(make_shared<tsf1nn::DAGen>(getter_tr_set, getter_cfe_set));
    // DTWFull
    gendist.push_back(make_shared<tsf1nn::DTWFullGen>(getter_tr_set, getter_cfe_set));
    // Wrap each distance generator in GenSplitter1NN (which is a i_GenNode) and push in generators
    for (auto const& gd : gendist) {
      generators.push_back(
        make_shared<tsf1nn::GenSplitterNN1>(gd, get_GenSplitterNN1_State, get_train_data, get_test_data)
      );
    }
  }

  // --- Put a node chooser over all generators
  auto node_gen = make_shared<tsf::node::meta::SplitterChooserGen>(std::move(generators), opt.nb_candidates);



  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Make PF and use it
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- ---
  // --- --- --- Build the forest TODO: make the forest, one tree for now
  // --- --- ---

  auto tree_trainer = std::make_shared<tsf::TreeTrainer>(leaf_gen, node_gen);

  // --- --- ---
  // --- --- --- TRAIN
  // --- --- ---

  auto train_start_time = utils::now();
  auto tree = tree_trainer->train(tstate, tdata, train_bcm);
  auto train_elapsed = utils::now() - train_start_time;

  // --- --- ---
  // --- --- --- TEST
  // --- --- ---

  classifier::ResultN result;
  auto test_start_time = utils::now();
  {
    const size_t test_size = test_dataset.size();
    for (size_t test_idx = 0; test_idx<test_size; ++test_idx) {
      classifier::Result1 r1 = tree->predict(tstate, tdata, test_idx);
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