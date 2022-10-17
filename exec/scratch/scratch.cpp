#include <iostream>
#include <tempo/utils/utils.hpp>
#include <tempo/distance/utils.hpp>
#include "tempo/distance/cost_functions.hpp"

using namespace std;
using namespace arma;

using F = double;

F softmin3(F a, F b, F c, F gamma) {

  a /= -gamma;
  b /= -gamma;
  c /= -gamma;

  const F max_val = std::max(std::max(a, b), c);

  F tmp = 0;
  tmp += exp(a - max_val);
  tmp += exp(b - max_val);
  tmp += exp(c - max_val);

  return -gamma*(log(tmp) + max_val);
}

F softmin3_(F a, F b, F c, F gamma) {

  a /= -gamma;
  b /= -gamma;
  c /= -gamma;

  F tmp = 0;
  tmp += exp(a);
  tmp += exp(b);
  tmp += exp(c);

  return -gamma*log(tmp);
}

F softdtw(const size_t lenI, const size_t lenJ,
          tempo::distance::utils::ICFun<F> auto cfun,
          const F gamma,
          const F cutoff,
          std::vector<F> buffer_v) {

  const size_t nblin = lenI + 1;
  const size_t nbcol = lenJ + 1;
  constexpr double DMAX = std::numeric_limits<double>::max();

  // std::vector<std::vector<F>> buffer;
  // buffer.resize(nblin, std::vector<F>(nbcol, DMAX));
  // auto buffat = [&buffer](size_t i, size_t j) -> F& { return buffer[i][j]; };

  buffer_v.assign(nblin*nbcol, DMAX);
  F *buffer = buffer_v.data();
  auto buffat = [buffer, nbcol](size_t i, size_t j) -> F& { return buffer[i*nbcol + j]; };

  buffat(0, 0) = 0;

  for (size_t i = 1; i<nblin; ++i) {
    F minv = DMAX;
    for (size_t j = 1; j<nbcol; ++j) {
      F d = cfun(i - 1, j - 1);
      F v = d + softmin3(buffat(i - 1, j), buffat(i - 1, j - 1), buffat(i, j - 1), gamma);
      minv = std::min<F>(minv, v);
      buffat(i, j) = v;
    }
    if(minv>cutoff){
      cout << "EA i=" << i << " minv = " << minv << " > " << cutoff << endl;
      return DMAX;
    }
  }

  return buffat(lenI, lenJ);
}

int main(int argc, char **argv) {

  std::cout << softmin3(5, 6, 7*10e-10, 1) << endl;

  std::vector<F> a{1, 2, 3, 4, 5};
  std::vector<F> b{1, 2, 3, 4, 5};

  constexpr auto sqed = tempo::distance::univariate::idx_ad2<F, std::vector<F>>;
  auto cfun = sqed(a, b);

  std::vector<F> buf;
  F res = softdtw(a.size(), b.size(), cfun, 1.0, -2, buf);
  std::cout << "softDTW = " << res << endl;

}

