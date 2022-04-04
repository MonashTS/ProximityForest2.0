#include <memory>
#include <libtempo/tseries/tseries.hpp>
#include <libtempo/reader/ts/ts.hpp>
#include <libtempo/classifier/proximity_forest/pftree.hpp>
#include <libtempo/classifier/proximity_forest/splitters.hpp>
#include <libtempo/classifier/proximity_forest/ipf.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/transform/derivative.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
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


    /*
    auto derives = transform::derive(train_dataset, 3);
    auto d1 = std::move(derives[0]);
    auto d2 = std::move(derives[1]);
    auto d3 = std::move(derives[2]);
    cout << d1.name() << endl;
    cout << d2.name() << endl;
    cout << d3.name() << endl;

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
    namespace pfs = pf::State;
    using F = double;
    using L = std::string;

    struct PFState :
      public pf::IStrain<L, PFState, PFState>,
      public pfs::PRNG_mt64,
      public pfs::TimeSeriesDataset<F, L>,
      public pfs::TimeSeriesDatasetHeader<PFState, F, L> {

      int depth{0};

      PFState(size_t seed, const pfs::TimeSeriesDataset<F, L>::MAP_t& transformations) :
        PRNG_mt64(seed),
        pfs::TimeSeriesDataset<double, std::string>(transformations) {}

      PFState clone(size_t /* bidx */) override {
        auto other = *this;
        other.depth += 1;
        std::cout << "CLONE DEPTH = " << other.depth << std::endl;
        return other;
      }

      void merge(PFState&& /* substate */) override {}
    };

    using Strain = PFState;
    using Stest = PFState;

    std::random_device rd;
    size_t seed = rd();
    auto transformations = pfs::TimeSeriesDataset<F, L>::MAP_t();
    transformations.insert({"default", train_dataset});

    std::vector<ByClassMap<L>> train_bcm{std::get<0>(train_dataset.header().get_BCM())};

    PFState train_state = PFState(seed, transformations);

    //std::unique_ptr<pf::IPF_NodeGenerator<std::string, PFState, PFState>>
    //sg = std::make_unique<pf::SG_1NN_DA<F,L,Strain,Stest>>("default", 1);

    auto sg_1nn_da_e1 = std::make_shared<pf::SG_1NN_DA<F, L, Strain, Stest>>("default", 1);
    auto sg_1nn_da_e2 = std::make_shared<pf::SG_1NN_DA<F, L, Strain, Stest>>("default", 2);
    auto sg_1nn_dtwf_e1 = std::make_shared<pf::SG_1NN_DTWFull<F, L, Strain, Stest>>("default", 1);
    auto sg_1nn_dtwf_e2 = std::make_shared<pf::SG_1NN_DTWFull<F, L, Strain, Stest>>("default", 2);

    auto sg_chooser = std::make_shared<pf::SG_chooser<L, Strain, Stest>>(
      pf::SG_chooser<L, Strain, Stest>::SGVec_t { sg_1nn_da_e1, sg_1nn_da_e2, sg_1nn_dtwf_e1, sg_1nn_dtwf_e2 },
      3 );

    auto sgleaf_purenode = std::make_shared<pf::SGLeaf_PureNode<F, L, Strain, Stest>>();

    pf::IPF_TopGenerator<L,Strain,Stest> sg_top(sgleaf_purenode, sg_chooser);

    auto tree = pf::PFTree<std::string, PFState>::make_node<PFState>(train_state, train_bcm, sg_top);
    auto classifier = tree->get_classifier();

    // --- --- ---
    auto test_transformations = pfs::TimeSeriesDataset<F, L>::MAP_t();
    test_transformations.insert({"default", test_dataset});

    size_t test_seed = rd();
    PFState test_state = PFState(test_seed, test_transformations);

    const size_t test_top = test_dataset.header().size();
    size_t correct = 0;
    for (size_t i = 0; i<test_top; ++i) {
      std::cout << i << std::endl;
      auto vec = classifier.predict_proba(test_state, i);
      size_t predicted_idx = std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
      std::string predicted_l = train_dataset.header().index_to_label().at(predicted_idx);
      if (predicted_l==test_dataset.header().labels()[i].value()) {
        correct++;
      } else {
        std::cout << "i predicted =" << predicted_l << " vs " << test_dataset.header().labels()[i].value() << std::endl;
      }
    }
    std::cout << "Correct = " << correct << "/" << test_top << std::endl;
  }

  return 0;
}
