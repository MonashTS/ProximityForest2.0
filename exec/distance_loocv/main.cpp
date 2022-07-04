#include <string>
#include <vector>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

int main(int argc, char **argv) {
  using namespace std;

  string program_name(*argv);
  vector<string> args(argv + 1, argv + argc);
  if(args.empty()){
    std::cout << "TODO: prin usage" << std::endl;
    exit(0);
  }

  return 0;
}
