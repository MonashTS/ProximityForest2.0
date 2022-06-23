#include "pch.h"
#include <tempo/utils/simplecli.hpp>
#include <tempo/reader/new_reader.hpp>
#include <tempo/distance/sliding/cross_correlation.hpp>

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
  if (code==0) {
    if (msg) { std::cout << msg.value() << std::endl; }
  } else {
    std::cerr << "<no usage defined>" << std::endl;
    if (msg) { std::cerr << msg.value() << std::endl; }
  }
  exit(code);
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace tempo;
  using namespace tempo::utils;

  // --- --- --- --- --- ---
  // Program Arguments
  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if (args.empty()) { return 0; }

  // Path to UCR
  fs::path UCRPATH;
  {
    auto parg_UCRPATH = tempo::scli::get_parameter<string>(args, "-p", tempo::scli::extract_string);
    if (!parg_UCRPATH) { do_exit(1, "specify the UCR path with the -p flag, e.g. -p:/path/to/dataset"); }
    UCRPATH = fs::path(parg_UCRPATH.value());
  }

  // Dataset name
  string dsname;
  {
    auto parg_UCRNAME = tempo::scli::get_parameter<string>(args, "-n", tempo::scli::extract_string);
    if (!parg_UCRNAME) { do_exit(1, "specify the UCR dataset name with the -n flag -n:NAME"); }
    dsname = parg_UCRNAME.value();
  }


  // --- --- --- --- --- ---
  // Read the dataset
  fs::path dspath = UCRPATH/dsname;

  DataSplit <TSeries> train_split = std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TRAIN.ts"), "train"));
  DatasetHeader const& train_header = train_split.header();
  size_t train_top = train_header.size();

  DataSplit <TSeries> test_split =
    std::get<1>(reader::load_dataset_ts(dspath/(dsname + "_TEST.ts"), "test", train_header.label_encoder()));
  DatasetHeader const& test_header = test_split.header();
  size_t test_top = test_header.size();


  // --- --- --- --- --- ---
  // Test your stuff here
  /*
  tempo::ByClassMap bcm = std::get<0>(train_split.get_BCM());
  IndexSet IS_0 = (*bcm).begin()->second;
  for(const auto& i: IS_0){ std::cout << i << endl; }

  auto const& c0_0 = train_split.at(IS_0[0]);
  auto const& c0_1 = train_split.at(IS_0[1]);
  auto const& c0_2 = train_split.at(IS_0[2]);
  auto const& c0_3 = train_split.at(IS_0[3]);

  auto const& other = train_split.at(7);

  cout << tempo::distance::SBD(c0_0.rowvec(), c0_0.rowvec()) << endl;
  cout << tempo::distance::SBD(c0_0.rowvec(), c0_1.rowvec()) << endl;
  cout << tempo::distance::SBD(c0_0.rowvec(), c0_2.rowvec()) << endl;
  cout << tempo::distance::SBD(c0_0.rowvec(), c0_3.rowvec()) << endl;

  cout << tempo::distance::SBD(c0_0.rowvec(), other.rowvec()) << endl;
  cout << tempo::distance::SBD(c0_0.rowvec(), other.rowvec()) << endl;
  cout << tempo::distance::SBD(c0_0.rowvec(), other.rowvec()) << endl;
  cout << tempo::distance::SBD(c0_0.rowvec(), other.rowvec()) << endl;
   */

  arma::rowvec r0{0, 1, 2, 3, 0, 1, 2, 3};
  arma::cx_rowvec fft0 = arma::fft(r0);
  arma::rowvec crosscorel0 = arma::real(arma::ifft(fft0 % arma::conj(fft0)));

  r0.print("r0");
  fft0.print("fft0");
  crosscorel0.print("cc0");

  std::cout << std::endl;

  arma::rowvec r1{1, 1, 1, 1, 1, 1, 1};
  arma::cx_rowvec fft1 = arma::fft(r1);
  r1.print("r1");
  fft1.print("fft1");


}
