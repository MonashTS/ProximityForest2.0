#include <memory>
#include <libtempo/tseries/tseries.hpp>
#include <libtempo/reader/ts/ts.hpp>
#include <libtempo/classifier/splitting_forest/proximity_forest/pf2018.hpp>
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
constexpr bool B = true;

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

  namespace pf = libtempo::classifier::pf;
  using F = double;
  using L = std::string;

  // --- --- --- Reading of a time series
  if (argc!=3) {
    std::cerr << "bar args: must be <path to uce folder> <dataset name>" << std::endl;
    exit(5);
  }

  // --- --- --- Read dataset
  std::string strpath(argv[1]);
  std::string name(argv[2]);
  fs::path dirpath(strpath + "/" + name);

  fs::path adiac_train = dirpath/(name + "_TRAIN.ts");
  auto train_dataset = load_dataset(adiac_train);

  fs::path adiac_test = dirpath/(name + "_TEST.ts");
  auto test_dataset = load_dataset(adiac_test);

  auto train_derives = transform::derive(train_dataset, 2);
  auto train_d1 = std::move(train_derives[0]);
  auto train_d2 = std::move(train_derives[1]);
  cout << train_d1.name() << endl;
  cout << train_d2.name() << endl;

  auto test_derives = transform::derive(test_dataset, 2);
  auto test_d1 = std::move(test_derives[0]);
  auto test_d2 = std::move(test_derives[1]);
  cout << test_d1.name() << endl;
  cout << test_d2.name() << endl;

  // --------------------------------------------------------------------------------------------------------------


  size_t nbt = 100;
  size_t nbc = 5;
  size_t nbthread = 8;

  pf::PF2018<L> pf2018(nbt, nbc);



  // --------------------------------------------------------------------------------------------------------------

  std::random_device rd;
  size_t seed = rd();

  auto transformations = std::make_shared<classifier::pf::DatasetMap_t<F, L>>();
  transformations->insert({"default", train_dataset});
  transformations->insert({"d1", train_d1});
  transformations->insert({"d2", train_d2});

  auto test_transformations = std::make_shared<classifier::pf::DatasetMap_t<F, L>>();
  test_transformations->insert({"default", test_dataset});
  test_transformations->insert({"d1", test_d1});
  test_transformations->insert({"d2", test_d2});
  size_t test_seed = seed + 5;

  auto trained_forest = pf2018.train(seed, transformations, nbthread);
  auto classifier = trained_forest.get_classifier_for(test_seed, test_transformations, B);

  if constexpr(B) {

    std::vector<size_t> depths;
    pf::DistanceSplitterState<L> distance_splitter_state(B);

    for (const auto& s : trained_forest.trained_states) {
      distance_splitter_state.merge(s->distance_splitter_state);
      depths.push_back(s->max_depth);
    }

    // Report on selected distances
    for (const auto&[n, c] : distance_splitter_state.selected_distances) {
      std::cout << n << ": " << c << std::endl;
    }
    // Report on depth
    double ad = 0;
    for (auto d : depths) {
      std::cout << d << "  ";
      ad += d;
    }
    std::cout << endl;
    std::cout << "Average depth = " << ad/depths.size() << std::endl;
  }

  const size_t test_top = test_dataset.header().size();
  size_t correct = 0;

  for (size_t i = 0; i<test_top; ++i) {
    auto[weight, proba] = classifier.predict_proba(i, nbthread);
    size_t predicted_idx = std::distance(proba.begin(), std::max_element(proba.begin(), proba.end()));
    std::string predicted_l = train_dataset.header().index_to_label().at(predicted_idx);
    std::string true_l = test_dataset.header().labels()[i].value();
    if (predicted_l==true_l) { correct++; }
    //std::cout << "Test instance " << i << " Weight: " << weight << " Proba:";
    //for (const auto p : proba) { std::cout << " " << p; }
    //std::cout << std::endl;
    //std::cout << "  Predicted index = " << predicted_idx;
    //std::cout << "  Predicted class = '" << predicted_l << "'";
    //std::cout << "  True class = '" << true_l << "'" << std::endl;
  }

  std::cout << "Result with " << nbt << " trees:" << std::endl;
  std::cout << "  correct: " << correct << "/" << test_top << std::endl;
  std::cout << "  accuracy: " << (double)correct/test_top << std::endl;

  return 0;
}







/* TODO:! move in test for derivcatve
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


/* Splitter list
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


// --- --- --- 1NN Elastic Distance Node Generator Default
auto sg_1nn_def_da =
  std::make_shared<pf::SG_1NN_DA<F, L, Strain, Stest>>(tname_def, exponents);

auto sg_1nn_def_adtw =
  std::make_shared<pf::SG_1NN_ADTW<F, L, Strain, Stest>>(tname_def, exponents, 2000, 20, 4);

auto sg_1nn_def_dtwf =
  std::make_shared<pf::SG_1NN_DTWFull<F, L, Strain, Stest>>(tname_def, exponents);

auto sg_1nn_def_dtw =
  std::make_shared<pf::SG_1NN_DTW<F, L, Strain, Stest>>(tname_def, exponents);

auto sg_1nn_def_wdtw =
  std::make_shared<pf::SG_1NN_WDTW<F, L, Strain, Stest>>(tname_def, exponents);

auto sg_1nn_def_erp =
  std::make_shared<pf::SG_1NN_ERP<F, L, Strain, Stest>>(tname_def, exponents);

auto sg_1nn_def_lcss =
  std::make_shared<pf::SG_1NN_LCSS<F, L, Strain, Stest>>(tname_def);

auto sg_1nn_def_msm =
  std::make_shared<pf::SG_1NN_MSM<F, L, Strain, Stest>>(tname_def, msm_costs);

auto sg_1nn_def_twe =
  std::make_shared<pf::SG_1NN_TWE<F, L, Strain, Stest>>(tname_def, twe_nus, twe_lambdas);


// --- --- --- 1NN Elastic Distance Node Generator D1
auto sg_1nn_d1_da =
  std::make_shared<pf::SG_1NN_DA<F, L, Strain, Stest>>(tname_d1, exponents);

auto sg_1nn_d1_adtw =
  std::make_shared<pf::SG_1NN_ADTW<F, L, Strain, Stest>>(tname_d1, exponents, 2000, 20, 4);

auto sg_1nn_d1_dtwf =
  std::make_shared<pf::SG_1NN_DTWFull<F, L, Strain, Stest>>(tname_d1, exponents);

auto sg_1nn_d1_dtw =
  std::make_shared<pf::SG_1NN_DTW<F, L, Strain, Stest>>(tname_d1, exponents);

auto sg_1nn_d1_wdtw =
  std::make_shared<pf::SG_1NN_WDTW<F, L, Strain, Stest>>(tname_d1, exponents);

auto sg_1nn_d1_erp =
  std::make_shared<pf::SG_1NN_ERP<F, L, Strain, Stest>>(tname_d1, exponents);

auto sg_1nn_d1_lcss =
  std::make_shared<pf::SG_1NN_LCSS<F, L, Strain, Stest>>(tname_d1);

auto sg_1nn_d1_msm =
  std::make_shared<pf::SG_1NN_MSM<F, L, Strain, Stest>>(tname_d1, msm_costs);

auto sg_1nn_d1_twe =
  std::make_shared<pf::SG_1NN_TWE<F, L, Strain, Stest>>(tname_d1, twe_nus, twe_lambdas);
*/


/*
auto sg_try_all = std::make_shared<pf::SG_try_all<L, Strain, Stest>>(
  pf::SG_try_all<L, Strain, Stest>::SGVec_t{
    sg_1nn_def_da,
    sg_1nn_def_dtw,
    sg_1nn_def_dtwf,
    sg_1nn_def_adtw,
    sg_1nn_def_wdtw,
    sg_1nn_def_erp,
    sg_1nn_def_lcss,
    sg_1nn_def_msm,
    sg_1nn_def_twe,
    sg_1nn_d1_da,
    sg_1nn_d1_dtw,
    sg_1nn_d1_dtwf,
    sg_1nn_d1_adtw,
    sg_1nn_d1_wdtw,
    sg_1nn_d1_erp,
    sg_1nn_d1_lcss,
    sg_1nn_d1_msm,
    sg_1nn_d1_twe,
  }
);





  // --- --- --- PF 2018
  auto sg_2018_chooser = libtempo::utils::initBlock {

    // --- --- --- Parameters ranges
    //auto exponents = std::make_shared<std::vector<double>>(std::vector<double>{0.25, 0.33, 0.5, 1, 2, 3, 4});
    auto exp2 = std::make_shared<std::vector<double>>(std::vector<double>{2});
    auto tnames = std::make_shared<std::vector<std::string>>(std::vector<std::string>{"default", "d1", "d2"});
    auto def = std::make_shared<std::vector<std::string>>(std::vector<std::string>{"default"});
    auto d1 = std::make_shared<std::vector<std::string>>(std::vector<std::string>{"d1"});
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

    /// TWE nu parameters
auto twe_nus = std::make_shared<std::vector<double>>(
  std::vector<double>{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1}
);

/// TWE lambda parameters
auto twe_lambdas = std::make_shared<std::vector<double>>(
  std::vector<double>{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667, 0.077777778,
                      0.088888889, 0.1}
);

// SQED
auto sg_1nn_da =
  std::make_shared<pf::SG_1NN_DA<F, L, Strain, Stest>>(def, exp2);

// DTW Full Window
auto sg_1nn_dtwf =
  std::make_shared<pf::SG_1NN_DTWFull<F, L, Strain, Stest>>(def, exp2);

// DDTW Full Window
auto sg_1nn_ddtwf =
  std::make_shared<pf::SG_1NN_DTWFull<F, L, Strain, Stest>>(d1, exp2);

// DTW Window
auto sg_1nn_dtw = std::make_shared<pf::SG_1NN_DTW<F, L, Strain, Stest>>(def, exp2);

// DDTW Window
auto sg_1nn_ddtw = std::make_shared<pf::SG_1NN_DTW<F, L, Strain, Stest>>(d1, exp2);

// WDTW
auto sg_1nn_wdtw = std::make_shared<pf::SG_1NN_WDTW<F, L, Strain, Stest>>(def, exp2);

// WDDTW
auto sg_1nn_wddtw = std::make_shared<pf::SG_1NN_WDTW<F, L, Strain, Stest>>(d1, exp2);

// ERP
auto sg_1nn_erp =
  std::make_shared<pf::SG_1NN_ERP<F, L, Strain, Stest>>(def, exp2);

// LCSS
auto sg_1nn_lcss =
  std::make_shared<pf::SG_1NN_LCSS<F, L, Strain, Stest>>(def);

// MSM
auto sg_1nn_msm =
  std::make_shared<pf::SG_1NN_MSM<F, L, Strain, Stest>>(def, msm_costs);

// TWE
auto sg_1nn_twe =
  std::make_shared<pf::SG_1NN_TWE<F, L, Strain, Stest>>(def, twe_nus, twe_lambdas);

return std::make_shared<pf::SG_chooser<L, Strain, Stest>>(
pf::SG_chooser<L, Strain, Stest>::SGVec_t({
sg_1nn_da,
sg_1nn_dtwf,
sg_1nn_ddtwf,
sg_1nn_dtw,
sg_1nn_ddtw,
sg_1nn_wdtw,
sg_1nn_wddtw,
sg_1nn_erp,
sg_1nn_lcss,
sg_1nn_msm,
sg_1nn_twe
}
), 5
);
};





















*/
