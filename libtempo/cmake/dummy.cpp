#include <libtempo/tseries/tseries.hpp>
#include <libtempo/distance/erp.hpp>

#include <vector>

using namespace std;

int main(int argc, char** argv){

  vector<double> a{0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
  vector<double> b{0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
  double gv = 0.1;

  auto dist_ = [](double a, double b){
    const double d = a-b;
    return d*d;
  };

  auto dist = [&](size_t i, size_t j){return dist_(a[i], b[j]);};
  auto gdist_li = [&, gv](size_t i){ return dist_(a[i], gv); };
  auto gdist_co = [&, gv](size_t i){ return dist_(gv, b[i]); };


}
