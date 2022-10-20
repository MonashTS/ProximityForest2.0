#include <tempo/distance/tseries.univariate.hpp>
#include <tempo/classifier/loocv/partable/dist_interface.hpp>

#include <nlohmann/json.hpp>

struct WDTW : tempo::classifier::nn1loocv::i_LOOCVDist {
  tempo::DTS train;
  tempo::DTS test;
  double cfe;

  using PType = std::tuple<size_t, double>;         // i, g
  std::vector<PType> params{};
  std::vector<std::vector<double>> p_weights{};

  // Updated by set_loocv_result
  std::vector<size_t> all_index{};
  std::vector<double> all_g{};
  double best_g{0};
  std::vector<double> best_weights;

  void generate_params() {
    constexpr size_t NBP = 100;
    const size_t maxl = train.header().length_max();
    // Generate weights
    for (size_t i = 0; i<=NBP; ++i) {
      double g = (double)i/(double)NBP;
      params.emplace_back(std::tuple(i, g));
      p_weights.emplace_back(tempo::distance::univariate::wdtw_weights(g, maxl));
    }
    // WDTW is similar to DTW: smaller g leads to higher cost
    // Reverse param for LOOCV and fixup indexes
    std::reverse(params.begin(), params.end());
    std::reverse(p_weights.begin(), p_weights.end());
    for (size_t i = 0; i<params.size(); ++i) { std::get<0>(params.at(i)) = i; }
    nb_params = params.size();
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Constructor / Destructor
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  WDTW(tempo::DTS train, tempo::DTS test, double cfe) :
    tempo::classifier::nn1loocv::i_LOOCVDist(0),
    train(std::move(train)),
    test(std::move(test)),
    cfe(cfe) { generate_params(); }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Interface implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  tempo::F distance_param(size_t train_idx1, size_t train_idx2, size_t param_idx, tempo::F bsf) override {
    auto const& weights = p_weights.at(param_idx);
    return tempo::distance::univariate::wdtw(train[train_idx1], train[train_idx2], cfe, weights.data(), bsf);
  }

  tempo::F distance_UB(size_t train_idx1, size_t train_idx2, size_t /* param_idx */) override {
    return tempo::distance::univariate::directa(train[train_idx1], train[train_idx2], cfe, tempo::utils::PINF);
  }

  tempo::F distance_test(size_t test_idx, size_t train_idx, tempo::F bsf) override {
    return tempo::distance::univariate::wdtw(test[test_idx], train[train_idx], cfe, best_weights.data(), bsf);
  }

  void set_loocv_result(std::vector<size_t> bestp) override {
    // Report all the best in order - extract r and omega for json report
    // Note: sort on tuples, where first component is size_t
    std::sort(bestp.begin(), bestp.end());
    for (const auto& pi : bestp) {
      auto [i, g] = params.at(pi);
      all_index.push_back(i);
      all_g.push_back(g);
      std::cout << "parameter " << i << " g = " << g << std::endl;
    }

    // Pick the smallest g, which is at the back of the array
    best_g = all_g.back();
    best_weights = p_weights[all_index.back()];
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // JSON
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  nlohmann::json to_json() {
    nlohmann::json j;
    j["name"] = "WDTW";
    j["best_indexes"] = tempo::utils::to_json(all_index);
    j["best_g"] = tempo::utils::to_json(all_g);
    j["selected_g"] = best_g;
    j["cost_function_exponent"] = cfe;
    return j;
  }

};
