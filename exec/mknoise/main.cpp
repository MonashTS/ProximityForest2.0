#include <tempo/reader/reader.hpp>
#include <tempo/writer/ts/ts.hpp>
#include <tempo/transform/tseries.univariate.hpp>
#include "tempo/reader/dts.reader.hpp"

#include <random>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char **argv) {

  std::random_device rd;
  tempo::PRNG prng(rd());

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // CMDLine
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::string program_name(*argv);
  std::vector<std::string> args(argv + 1, argv + argc);
  if (argc!=5) {
    std::cout << "<path to ucr> <dataset name> <delta> <output dir root> required" << std::endl;
    exit(1);
  }
  std::vector<std::string> argList(argv, argv + argc);
  std::filesystem::path path_ucr(argList[1]);
  std::string dataset_name(argList[2]);
  double delta = std::stod(argList[3]);
  std::filesystem::path outdirroot(argList[4]);

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Read dataset
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  tempo::DTS train_dataset;
  tempo::DTS test_dataset;
  {
    using namespace tempo::reader::dataset;
    Result read_dataset_result;
    ts_ucr conf;
    conf.ucr_dir = path_ucr;
    conf.name = dataset_name;

    read_dataset_result = load({conf});

    if (read_dataset_result.index()==0) {
      std::cout << "Error reading dataset" << std::endl;
      exit(1);
    }

    TrainTest traintest = std::get<1>(std::move(read_dataset_result));
    train_dataset = traintest.train_dataset;
    test_dataset = traintest.test_dataset;

    // --- --- --- Sanity check
    std::vector<std::string> errors = sanity_check(traintest);

    if (!errors.empty()) {
      std::cout << "Sanity check failed" << std::endl;
      exit(0);
    }
  } // End of dataset loading


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Do noise
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  auto train_noise_transform =
    train_dataset.transform().map_shptr<tempo::TSeries>(
      [&](tempo::TSeries const& t) { return tempo::transform::univariate::noise(t, delta, prng); },
      "noise"
    );

  tempo::DTS train_noise("train", train_noise_transform);

  auto test_noise_transform =
    test_dataset.transform().map_shptr<tempo::TSeries>(
        [&](tempo::TSeries const& t) { return tempo::transform::univariate::noise(t, delta, prng); },
        "noise"
      );

  tempo::DTS test_noise("test", test_noise_transform);


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Output
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  std::filesystem::path outdir = outdirroot / dataset_name;
  std::filesystem::create_directories(outdir);
  std::ofstream out_train(outdir/(dataset_name+"_TRAIN.ts"), std::ios::out | std::ios::trunc);
  std::ofstream out_test(outdir/(dataset_name+"_TEST.ts"), std::ios::out | std::ios::trunc);

  tempo::univariate::writer::write(train_noise, dataset_name, out_train);
  tempo::univariate::writer::write(test_noise, dataset_name, out_test);

  return 0;

}
