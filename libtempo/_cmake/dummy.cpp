#include <libtempo/tseries/tseries.hpp>
#include <libtempo/reader/ts/ts.hpp>
#include <libtempo/classifier/proximity_forest/pftree.hpp>
#include <libtempo/classifier/proximity_forest/splitters.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/transform/derivative.hpp>


#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace fs = std::filesystem;
using namespace std;
using namespace libtempo;
using PRNG = std::mt19937_64;

void derivative(const double* series, size_t length, double* out) {
  if (length>2) {
    for (size_t i{1}; i<length-1; ++i) {
      out[i] = ((series[i]-series[i-1])+((series[i+1]-series[i-1])/2.0))/2.0;
    }
    out[0] = out[1];
    out[length-1] = out[length-2];
  } else {
    std::copy(series, series+length, out);
  }
}

int main(int argc, char** argv) {

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
    for (const auto& i: row) {
      cout << i << " ";
    }
    cout << endl;
  }

  // --- --- --- Reading of a time series
  if (argc>1) {
    std::string strpath(argv[1]);
    fs::path adiac_train(strpath);
    std::ifstream istream(adiac_train);

    auto res = libtempo::reader::TSReader::read(istream);
    if (res.index()==0) {
      cerr << "reading error: " << get<0>(res) << endl;
      return 1;
    }

    auto dataset = std::move(get<1>(res));
    cout << dataset.name() << endl;
    cout << "Has missing value: " << dataset.has_missing_value() << endl;

    auto derives = transform::derive(dataset, 3);
    auto d1 = std::move(derives[0]);
    auto d2 = std::move(derives[1]);
    auto d3 = std::move(derives[2]);
    cout << d1.name() << endl;
    cout << d2.name() << endl;
    cout << d3.name() << endl;

    for (size_t i = 0; i<dataset.size(); ++i) {

      const auto& ts = dataset.data()[i];
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

  }

  using namespace libtempo::classifier::pf;

  struct State {

    std::optional<std::string> get_label(size_t i) {
      std::string l = std::to_string(i%2);
      return {l};
    }

    decltype(&weighted_gini_impurity<std::string>) split_evaluator = weighted_gini_impurity;

  };


  // Make "fake data" for the tree
  std::shared_ptr<State> st = std::make_shared<State>();
  IndexSet is(20);
  ByClassMap<std::string> bcm;
  for (const auto i: is) {
    const auto label = st->get_label(i).value();
    bcm[label].push_back(i);
  }

  SplitterGenerator<std::string, State, PRNG>
    generator{
    .generate = [](std::shared_ptr<State> state, const IndexSet& is, const ByClassMap<std::string>& bcm, PRNG& prng) {
      Splitter_uptr<std::string, State, PRNG> res;
      const auto idx = std::uniform_int_distribution<size_t>(0, is.size()-1)(prng);
      if (idx<=is.size()/4) {
        // Generate a "good" splitter
        auto my_state = std::make_shared<size_t>(idx);
        res = std::make_unique<Splitter<std::string, State, PRNG>>(Splitter<std::string, State, PRNG>{
          .train=[my_state](std::shared_ptr<State>& state, const IndexSet& is, const ByClassMap<std::string>& bcm, PRNG& prng) {
            std::cout << "train good" << std::endl;
          },
          .classify_train=[my_state](std::shared_ptr<State>& state, size_t index, PRNG& prng) {
            std::cout << "train with: " << *my_state << std::endl;
            return state->get_label(index).value();
          },
          .classify_test=[my_state](std::shared_ptr<State>& state, size_t index, PRNG& prng) {
            return state->get_label(index).value();
          }
        });
      } else {
        // Generate a "bad" classifier
        res = std::make_unique<Splitter<std::string, State, PRNG>>(Splitter<std::string, State, PRNG>{
          .train=[](std::shared_ptr<State>& state, const IndexSet& is, const ByClassMap<std::string>& bcm, PRNG& prng) {
            std::cout << "train bad" << std::endl;
          },
          .classify_train=[](std::shared_ptr<State>& state, size_t index, PRNG& prng) {
            return std::to_string(std::uniform_int_distribution<>(0, 1)(prng));
          },
          .classify_test=[](std::shared_ptr<State>& state, size_t index, PRNG& prng) {
            return std::to_string(std::uniform_int_distribution<>(0, 1)(prng));
          }
        });
      }

      return res;
    }
  };

  std::random_device rd;
  PRNG prng(rd());

  auto tree = libtempo::classifier::pf::PFTree<std::string, State, PRNG>::make_tree(st, is, bcm, 1, generator, prng);

  std::cout << tree->is_pure_node << std::endl;
  auto treecl = tree->get_classifier(prng);
  for (int i = 0; i<20; ++i) {
    std::cout << "Classify " << i << " as " << treecl.classify(st, i) << std::endl;
  }

  return 0;
}
