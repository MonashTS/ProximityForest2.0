#include <tempo/utils/utils/stats.hpp>
#include <tempo/distance/tseries.univariate.hpp>
#include <tempo/classifier/loocv/partable/dist_interface.hpp>

#include <nlohmann/json.hpp>

struct ADTW : tempo::classifier::nn1loocv::i_LOOCVDist {
  static constexpr size_t SAMPLE_SIZE = 4000;
  static constexpr double penalty_exponent = 5.0;

  tempo::DTS train;
  tempo::DTS test;
  double cfe;

  double sample_mean_diagdist{0};

  using PType = std::tuple<size_t, double>;         // i, omega
  std::shared_ptr<std::vector<PType>> params;

  // Updated by set_loocv_result
  std::vector<size_t> all_index{};
  std::vector<double> all_omega{};
  double median_penalty{0};

  void do_sampling(tempo::PRNG& prng) {
    const int train_size = (int)train.size();
    tempo::utils::StddevWelford welford;
    std::uniform_int_distribution<> distrib(0, train_size - 1);
    for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
      const auto& q = train[distrib(prng)];
      const auto& s = train[distrib(prng)];
      const double cost = tempo::distance::univariate::directa(q, s, cfe, tempo::utils::PINF);
      welford.update(cost);
    }
    sample_mean_diagdist = welford.get_mean();
  }

  void generate_params() {
    // 0-98 : sampling
    for (size_t i = 0; i<99; ++i) {
      const double r = std::pow((double)i/100.0, penalty_exponent);
      const double omega = r*sample_mean_diagdist;
      params->emplace_back(std::tuple{i, omega});
    }
    // 99: PINF
    params->emplace_back(std::tuple{99, tempo::utils::PINF});
    // Check
    this->nb_params = params->size();
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Constructor / Destructor
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  ADTW(tempo::DTS train, tempo::DTS test, double cfe, tempo::PRNG& prng) :
    tempo::classifier::nn1loocv::i_LOOCVDist(0),
    train(std::move(train)),
    test(std::move(test)),
    cfe(cfe),
    params(std::make_shared<std::vector<PType>>()) {
    do_sampling(prng);
    generate_params();
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Interface implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  tempo::F distance_param(size_t train_idx1, size_t train_idx2, size_t param_idx, tempo::F bsf) override {
    const double penalty = std::get<1>(params->at(param_idx));
    return tempo::distance::univariate::adtw(train[train_idx1], train[train_idx2], cfe, penalty, bsf);
  }

  tempo::F distance_UB(size_t train_idx1, size_t train_idx2, size_t /* param_idx */) override {
    return tempo::distance::univariate::directa(train[train_idx1], train[train_idx2], cfe, tempo::utils::PINF);
  }

  tempo::F distance_test(size_t test_idx, size_t train_idx, tempo::F bsf) override {
    return tempo::distance::univariate::adtw(test[test_idx], train[train_idx], cfe, median_penalty, bsf);
  }

  void set_loocv_result(std::vector<size_t> bestp) override {
    // Report all the best in order - extract r and omega for json report
    // Note: sort on tuples, where first component is size_t
    std::sort(bestp.begin(), bestp.end());
    for (const auto& pi : bestp) {
      auto [i, omega] = params->at(pi);
      all_index.push_back(i);
      all_omega.push_back(omega);
      std::cout << "parameter " << i << " penalty = " << omega << std::endl;
    }

    // Pick the median as "the best"
    {
      auto size = bestp.size();
      if (size%2==0) { // Median "without a middle" = average
        const double omega1 = std::get<1>(params->at(bestp[size/2 - 1]));
        const double omega2 = std::get<1>(params->at(bestp[size/2]));
        median_penalty = (omega1 + omega2)/2.0;
      } else { // Median "middle"
        median_penalty = std::get<1>(params->at(bestp[size/2]));
      }
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // JSON
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  nlohmann::json to_json() {
    nlohmann::json j;
    j["name"] = "ADTW";
    j["sample_size"] = SAMPLE_SIZE;
    j["sample_value"] = sample_mean_diagdist;
    j["penalty_exponent"] = penalty_exponent;
    j["best_indexes"] = tempo::utils::to_json(all_index);
    j["best_penalties"] = tempo::utils::to_json(all_omega);
    j["median_penalty"] = median_penalty;
    j["cost_function_exponent"] = cfe;
    return j;
  }

};
