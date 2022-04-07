#include <memory>
#include <libtempo/tseries/tseries.hpp>
#include <libtempo/reader/ts/ts.hpp>
#include <libtempo/classifier/proximity_forest/pftree.hpp>
#include <libtempo/classifier/proximity_forest/splitters.hpp>
#include <libtempo/classifier/proximity_forest/splitters/distance_splitters.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/transform/derivative.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>

namespace fs = std::filesystem;
using namespace std;
using namespace libtempo;
using PRNG = std::mt19937_64;

void derivative(const double *series, size_t length, double *out) {
  if (length>2) {
    for (size_t i{1}; i<length - 1; ++i) {
      out[i] = ((series[i] - series[i - 1]) + ((series[i + 1] - series[i - 1])/2.0))/2.0;
    }
    out[0] = out[1];
    out[length - 1] = out[length - 2];
  } else {
    std::copy(series, series + length, out);
  }
}

auto load_dataset(const fs::path& path) {

  std::ifstream istream_(path);

  auto _res = libtempo::reader::TSReader::read(istream_);
  if (_res.index()==0) {
    cerr << "reading error: " << get<0>(_res) << endl;
    exit(1);
  }

  auto _dataset = std::move(get<1>(_res));
  cout << _dataset.name() << endl;
  cout << "Has missing value: " << _dataset.header().has_missing_value() << endl;

  return _dataset;
}

int main(int argc, char **argv) {

  vector<double> a{0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
  vector<double> b{1, NAN, 3, 4, 5, 6};

  TSeries t1 = TSeries<double, string>::mk_rowmajor(move(a), 1, "a", false);
  TSeries t2 = TSeries<double, string>::mk_rowmajor(move(b), 2, "a", {});

  cout << "t1 has nan " << t1.missing() << endl;
  cout << "t2 has nan " << t2.missing() << endl;
  cout << endl;
  cout << "t1 access (1) through row major map: " << t1.rm_emap()(1) << endl;
  cout << "t1 access (1) through col major map: " << t1.cm_emap()(1) << endl;
  cout << endl;
  cout << "t2 access (1,2) through row major map: " << t2.rm_emap()(1, 2) << endl;
  cout << "t2 access (1,2) through col major map: " << t2.cm_emap()(1, 2) << endl;
  cout << endl;
  cout << "t2 access [3] through row major ptr: " << t2.rm_data()[3] << endl;
  cout << "t2 access [3] through col major ptr: " << t2.cm_data()[3] << endl;
  cout << endl;
  cout << "t2 row by row" << endl;

  const auto& t2cmm = t2.cm_emap();
  for (auto r = 0; r<t2cmm.rows(); ++r) {
    const auto& row = t2cmm.row(r);
    for (const auto& i : row) {
      cout << i << " ";
    }
    cout << endl;
  }

  IndexSet is;
  for (auto i : is) {
    std::cout << "is = " << i << std::endl;
  }

  // --- --- --- Reading of a time series
  if (argc>2) {
    std::string strpath(argv[1]);
    std::string name(argv[2]);
    fs::path dirpath(strpath + "/" + name);

    fs::path adiac_train = dirpath/(name + "_TRAIN.ts");
    auto train_dataset = load_dataset(adiac_train);

    fs::path adiac_test = dirpath/(name + "_TEST.ts");
    auto test_dataset = load_dataset(adiac_test);

    auto derives = transform::derive(train_dataset, 2);
    auto d1 = std::move(derives[0]);
    auto d2 = std::move(derives[1]);
    cout << d1.name() << endl;
    cout << d2.name() << endl;

    auto test_derives = transform::derive(test_dataset, 2);
    auto test_d1 = std::move(test_derives[0]);
    auto test_d2 = std::move(test_derives[1]);
    cout << test_d1.name() << endl;
    cout << test_d2.name() << endl;

    /*
    for (size_t i = 0; i<train_dataset.size(); ++i) {

      const auto& ts = train_dataset.data()[i];
      const auto& tsd1 = d1.data()[i].rm_data();
      const auto& tsd2 = d2.data()[i].rm_data();
      const auto& tsd3 = d3.data()[i].rm_data();

      std::vector<double> v1(ts.size());
      std::vector<double> v2(ts.size());
      std::vector<double> v3(ts.size());

      derivative(ts.rm_data(), ts.size(), v1.data());
      derivative(v1.data(), ts.size(), v2.data());
      derivative(v2.data(), ts.size(), v3.data());

      for (size_t j = 0; j<ts.size(); ++j) {
        if (tsd1[j]!=v1[j]) {
          std::cerr << "ERROR D1 (i,j) = (" << i << ", " << j << ")" << std::endl;
          std::cerr << "TS size = " << ts.size() << std::endl;
          exit(5);
        }
        if (tsd2[j]!=v2[j]) {
          std::cerr << "ERROR D2 (i,j) = (" << i << ", " << j << ")" << std::endl;
          std::cerr << "TS size = " << ts.size() << std::endl;
          exit(5);
        }
        if (tsd3[j]!=v3[j]) {
          std::cerr << "ERROR D3 (i,j) = (" << i << ", " << j << ")" << std::endl;
          std::cerr << "TS size = " << ts.size() << std::endl;
          exit(5);
        }
      }
    }
     */



    // --------------------------------------------------------------------------------------------------------------
    namespace pf = libtempo::classifier::pf;
    using F = double;
    using L = std::string;

    struct PFState :
      public pf::IStrain<L, PFState, PFState>,
      public pf::TimeSeriesDatasetHeader<PFState, F, L> {

      std::map<std::string, size_t> selected_distances;

      size_t depth{0};

      /// Pseudo random number generator: use a unique pointer (stateful)
      std::unique_ptr<PRNG> prng;

      /// Dictionary of name->dataset of time series
      std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map;

      PFState(size_t seed, std::shared_ptr<pf::DatasetMap_t<F, L>> dataset_shared_map)
        : prng(std::make_unique<PRNG>(seed)), dataset_shared_map(std::move(dataset_shared_map)) {}

      /// Ensure we do not copy the state by error: we have to properly deal with the random number generator
      PFState(const PFState&) = delete;

    private:

      /// Cloning Constructor: create a new PRNG
      PFState(size_t depth, size_t new_seed, std::shared_ptr<pf::DatasetMap_t<F, L>> map) :
        depth(depth), prng(std::make_unique<PRNG>(new_seed)), dataset_shared_map(std::move(map)) {}

      /// Forking Constructor: transmit PRNG into the new state
      PFState(size_t depth, std::unique_ptr<PRNG>&& m_prng, std::shared_ptr<pf::DatasetMap_t<F, L>> map) :
        depth(depth), prng(std::move(m_prng)), dataset_shared_map(std::move(map)) {}

    public:

      /// Transmit the prng down the branch
      PFState branch_fork(size_t /* bidx */) override {
        return PFState(depth + 1, std::move(prng), dataset_shared_map);
      }

      /// Merge "other" into "this". Move the prng into this. Merge statistics into this
      void branch_merge(PFState&& other) override {
        prng = std::move(other.prng);
        for (const auto&[n, c] : other.selected_distances) {
          selected_distances[n] += c;
        }

      }

      /// Clone at the forest level - clones must be fully independent as they can be used in parallel
      /// Create a new prng
      std::unique_ptr<PFState> forest_clone() override {
        size_t new_seed = (*prng)();
        return std::unique_ptr<PFState>(new PFState(depth, new_seed, dataset_shared_map));
      }

      /// Merge in this a state that has been produced by forest_clone
      void forest_merge(std::unique_ptr<PFState> other) override {
        for (const auto&[n, c] : other->selected_distances) {
          selected_distances[n] += c;
        }
      }

    };







    using Strain = PFState;
    using Stest = PFState;

    std::random_device rd;
    size_t seed = rd();

    auto transformations = std::make_shared<classifier::pf::DatasetMap_t<F, L>>();
    transformations->insert({"default", train_dataset});
    transformations->insert({"d1", d1});
    transformations->insert({"d2", d2});
    seed = 15;
    PFState train_state = PFState(seed, transformations);

    auto test_transformations = std::make_shared<classifier::pf::DatasetMap_t<F, L>>();
    test_transformations->insert({"default", test_dataset});
    test_transformations->insert({"d1", test_d1});
    test_transformations->insert({"d2", test_d2});
    size_t test_seed = seed + 5;
    PFState test_state = PFState(test_seed, test_transformations);


    // --- --- --- Parameters range
    auto exponents = std::make_shared<std::vector<double>>(std::vector<double>{0.25, 0.33, 0.5, 1, 2, 3, 4});
    auto tnames = std::make_shared<std::vector<std::string>>(std::vector<std::string>{"default", "d1"}); // , "d2"});
    auto msm_costs = std::make_shared<std::vector<double>>(
      std::vector<double>{
        0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375,
        0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125,
        0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352,
        0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784,
        0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88,
        4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28,
        9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8,
        60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100
      }
    );

    /** TWE nu parameters */
    auto twe_nus = std::make_shared<std::vector<double>>(
      std::vector<double>{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1}
    );

    /** TWE lambda parameters */
    auto twe_lambdas = std::make_shared<std::vector<double>>(
      std::vector<double>{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667, 0.077777778,
                          0.088888889, 0.1}
    );

    // --- --- --- 1NN Elastic Distance Node Generator
    auto sg_1nn_da =
      std::make_shared<pf::SG_1NN_DA<F, L, Strain, Stest>>(tnames, exponents);

    auto sg_1nn_adtw =
      std::make_shared<pf::SG_1NN_ADTW<F, L, Strain, Stest>>(tnames, exponents, 2000, 20, 4);

    auto sg_1nn_dtwf =
      std::make_shared<pf::SG_1NN_DTWFull<F, L, Strain, Stest>>(tnames, exponents);

    auto sg_1nn_dtw =
      std::make_shared<pf::SG_1NN_DTW<F, L, Strain, Stest>>(tnames, exponents);

    auto sg_1nn_wdtw =
      std::make_shared<pf::SG_1NN_WDTW<F, L, Strain, Stest>>(tnames, exponents);

    auto sg_1nn_erp =
      std::make_shared<pf::SG_1NN_ERP<F, L, Strain, Stest>>(tnames, exponents);

    auto sg_1nn_lcss =
      std::make_shared<pf::SG_1NN_LCSS<F, L, Strain, Stest>>(tnames);

    auto sg_1nn_msm =
      std::make_shared<pf::SG_1NN_MSM<F, L, Strain, Stest>>(tnames, msm_costs);

    auto sg_1nn_twe =
      std::make_shared<pf::SG_1NN_TWE<F, L, Strain, Stest>>(tnames, twe_nus, twe_lambdas);

    auto sg_chooser = std::make_shared<pf::SG_chooser<L, Strain, Stest>>(
      pf::SG_chooser<L, Strain, Stest>::SGVec_t{
        //sg_1nn_da,
        sg_1nn_dtw,
        //sg_1nn_dtwf,
        sg_1nn_adtw,
        //sg_1nn_wdtw,
        //sg_1nn_erp,
        sg_1nn_lcss,
        //sg_1nn_msm,
        //sg_1nn_twe,
      }
      , 5
    );

    // --- --- --- Leaf Generator
    auto sgleaf_purenode = std::make_shared<pf::SGLeaf_PureNode<F, L, Strain, Stest>>();


    // --- --- --- Tree Trainer: made of a leaf generator (pure node) and a node generator (chooser)
    auto tree_trainer = std::make_shared<pf::PFTreeTrainer<L, Strain, Stest>>(sgleaf_purenode, sg_chooser);
    {
      std::vector<ByClassMap<L>> train_bcm{std::get<0>(train_dataset.header().get_BCM())};
      auto tree = tree_trainer->train(train_state, train_bcm);
      const size_t test_top = test_dataset.header().size();
      size_t correct = 0;
      for (size_t i = 0; i<test_top; ++i) {
        std::cout << i << std::endl;
        auto vec = tree->predict_proba(test_state, i);
        size_t predicted_idx = std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
        std::string predicted_l = train_dataset.header().index_to_label().at(predicted_idx);
        if (predicted_l==test_dataset.header().labels()[i].value()) {
          correct++;
        }
      }
      std::cout << "1 tree: correct = " << correct << "/" << test_top << std::endl;
      for (const auto&[n, c] : train_state.selected_distances) {
        std::cout << n << ": " << c << std::endl;
      }
      train_state.selected_distances.clear();
    }

    // --- --- --- Forest Trainer
    pf::PForestTrainer<L, Strain, Stest> forest_trainer(tree_trainer, 100);
    {
      std::vector<ByClassMap<L>> train_bcm{std::get<0>(train_dataset.header().get_BCM())};
      auto forest = forest_trainer.train(train_state, train_bcm);
      const size_t test_top = test_dataset.header().size();
      size_t correct = 0;
      for (size_t i = 0; i<test_top; ++i) {
        std::cout << i << std::endl;
        auto vec = forest->predict_proba(test_state, i);
        size_t predicted_idx = std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
        std::string predicted_l = train_dataset.header().index_to_label().at(predicted_idx);
        if (predicted_l==test_dataset.header().labels()[i].value()) {
          correct++;
        }
      }
      std::cout << "5 trees: correct = " << correct << "/" << test_top << std::endl;
      for (const auto&[n, c] : train_state.selected_distances) {
        std::cout << n << ": " << c << std::endl;
      }
      train_state.selected_distances.clear();
    }

  }

  return 0;
}
