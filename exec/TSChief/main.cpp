#include <exception>
#include <regex>

#include <tempo/utils/readingtools.hpp>
#include <tempo/reader/new_reader.hpp>
#include <tempo/transform/tseries.univariate.hpp>
#include <tempo/classifier/TSChief/forest.hpp>

#include <json/json.h>
#include "cmdline.hpp"
#include "pfsplitters.hpp"

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (msg) { std::cerr << msg.value() << std::endl; }
  exit(code);
}

int main(int argc, char **argv) {

  // --- --- --- Type / namespace
  using namespace std;
  using namespace tempo;
  using MDTS = std::map<std::string, tempo::DTS>;
  namespace tsc = tempo::classifier::TSChief;
  namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

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


  // --- --- ---
  // --- --- --- Constants & configurations
  // --- --- ---

  // --- Time series transforms
  const std::string tr_default("default");
  const std::string tr_d1("derivative1");
  // const std::string tr_d2("derivative2");

  // Make map of transforms (transform name, dataset)
  using MDTS = map<string, DTS>;
  shared_ptr<MDTS> train_map = make_shared<MDTS>();
  shared_ptr<MDTS> test_map = make_shared<MDTS>();

  auto prepare_data_start_time = utils::now();
  {
    namespace ttu = tempo::transform::univariate;

    // --- TRAIN
    auto train_derive_t1 = train_dataset.transform().map_shptr<TSeries>(ttu::derive, tr_d1);
    DTS train_derive_1("train", train_derive_t1);
    train_map->emplace("default", train_dataset);
    train_map->emplace(tr_d1, train_derive_1);
    // auto train_derive_t2 = train_derive_t1->map_shptr<TSeries>(ttu::derive, tr_d2);
    // DTS train_derive_2("train", train_derive_t2);
    // train_map->emplace(tr_d2, train_derive_2);

    // --- TEST
    auto test_derive_t1 = test_dataset.transform().map_shptr<TSeries>(ttu::derive, tr_d1);
    DTS test_derive_1("test", test_derive_t1);
    test_map->emplace("default", test_dataset);
    test_map->emplace(tr_d1, test_derive_1);
    // auto test_derive_t2 = test_derive_t1->map_shptr<TSeries>(ttu::derive, tr_d2);
    // DTS test_derive_2("test", test_derive_t2);
    // test_map->emplace(tr_d2, test_derive_2);
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


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Configure the splitters
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

   std::shared_ptr<tsc::i_GenLeaf> leaf_gen = pf2018::splitters::make_pure_leaf(get_train_header);
 //    std::shared_ptr<tsc::i_GenLeaf> leaf_gen = pf2018::splitters::make_pure_leaf_smoothp(get_train_header);


  std::shared_ptr<tsc::i_GenNode> node_gen;
  // --- temp for experiments
  std::vector<F> exponents{0.5, 1, 2};
  std::vector<std::string> transforms{tr_default, tr_d1};

  // extract distances in a set
  regex r(":");
  auto const& str = opt.pfconfig;
  std::set<std::string> distances(sregex_token_iterator(str.begin(), str.end(), r, -1), sregex_token_iterator());
  if (distances.empty()) {
    throw std::invalid_argument("No distances registered (" + str + ")");
  }

  opt.pfconfig = "PF";
  for (const auto& d : distances) {
    std::cout << "distance: " << d << std::endl;
    opt.pfconfig += ":" + d;
  }

  node_gen = pf2018::splitters::make_node_splitter(
    exponents, transforms, distances, opt.nb_candidates,
    train_header.length_max(),
    *train_map,
    get_train_data,
    get_test_data,
    tstate
  );


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Use the forest
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  auto tree_trainer = std::make_shared<tsc::TreeTrainer>(leaf_gen, node_gen);
  tsc::ForestTrainer forest_trainer(get_train_header, tree_trainer, opt.nb_trees);

  // --- --- --- TRAIN

  auto train_start_time = utils::now();
  auto forest = forest_trainer.train(tstate, tdata, train_bcm, opt.nb_threads, &std::cout);
  auto train_elapsed = utils::now() - train_start_time;

  // --- --- --- TEST

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
    j["nb_corrects"] = (int)nb_correct;
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