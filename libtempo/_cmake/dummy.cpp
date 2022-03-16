#include <libtempo/tseries/tseries.hpp>
#include <libtempo/reader/ts/ts.hpp>
#include <libtempo/classifier/proximity_forest/pftree.hpp>
#include <libtempo/tseries/dataset.hpp>


#include <vector>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace std;
using namespace libtempo;

int main(int argc, char** argv){

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
  if(argc>1) {
    std::string strpath(argv[1]);
    fs::path adiac_train(strpath);
    std::ifstream istream(adiac_train);
    auto res = libtempo::reader::TSReader::read(istream);
    if (res.index()==0) {
      cerr << "reading error: " << get<0>(res) << endl;
      return 1;
    }
    auto tsdata = std::move(get<1>(res));
    cout << tsdata.problem_name.value_or("no problem name specified") << endl;
    cout << "Has missing value: " << tsdata.missing.value() << endl;
  }


  // --- --- --- PF
  //libtempo::classifier::pf::PFNode<double, string> NODE;


  return 0;
}
