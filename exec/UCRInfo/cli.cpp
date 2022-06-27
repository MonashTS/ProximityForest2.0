
#include "cli.hpp"
#include "tempo/distance/elastic/adtw.hpp"

std::string usage =
  "UCRInfo\n"
  "Monash University, Melbourne, Australia 2022\n"
  "Dr. Matthieu Herrmann\n"
  "Compute information over a UCR dataset"
  "UCRInfo <-p:> <-n:> [-s:sampling] [-seed:] [-out:]\n"
  "Mandatory arguments:\n"
  "  -p:<path to the ucr archive folder>   e.g. '-p:/home/myuser/Univariate_ts'\n"
  "  -n:<name of the dataset>              e.g. '-n:Adiac' Must correspond to the dataset's folder name\n"
  "Optional arguments [with their default values]:\n"
  "  -modminkowski:<int n>:<double e>  Modified Minkowski distance with exponent <e> sampling over the each splits\n"
  "                                 Sample n distance Minkowski(a, b, e) (with a != b) over each split."
  "                                 n<=0 means no sampling (default)\n"
  "                                 Note: Modified Minkowski by not taking the e-th root of the result"
  "  -seed:<int n>   Fixed seed of randomness. Generate a random seed if n<0 [n = -1] !\n"
  "  -out:<path>     Where to write the json file. If the file exists, overwrite it."
  "";

[[noreturn]] void do_exit(int code, std::optional<std::string> msg) {
  if (code==0) {
    if (msg) { std::cout << msg.value() << std::endl; }
  } else {
    std::cerr << usage << std::endl;
    if (msg) { std::cerr << msg.value() << std::endl; }
  }
  exit(code);
}