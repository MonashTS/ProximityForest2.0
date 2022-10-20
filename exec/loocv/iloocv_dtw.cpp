#include <tempo/distance/tseries.univariate.hpp>
#include <tempo/classifier/loocv/partable/dist_interface.hpp>

#include <nlohmann/json.hpp>

struct DTW : tempo::classifier::nn1loocv::i_LOOCVDist {
  tempo::DTS train;
  tempo::DTS test;
  double cfe;

  using PType = std::tuple<size_t, size_t>;         // i, w
  std::shared_ptr<std::vector<PType>> params;

  // Updated by set_loocv_result
  std::vector<size_t> all_index{};
  std::vector<size_t> all_w{};
  size_t best_window{0};

  void generate_params() {
    constexpr size_t NBP = 100;
    const size_t maxw = train.header().length_max(); // -2; -// In theory -2, but for now, stick with EE

    // For DTW, we must have dtw(a, b, w0) <= dtw(a, b, w1), i.e. w0 >= w1.
    // I.e. param[0] is the largest window, and params[last] is the smallest one.
    // We want to have NBP+1 parameters (including window 0 and window max), but we may have less
    // Note that we generate the params from low window to high: we'll have to reverse the array and fixup the indexes
    params->emplace_back(std::tuple(0, 0));

    for(size_t i=1; i<=NBP; ++i){
      double ratio = (double) i / double(NBP);  // NBP-1: guaranty we hit 100%
      size_t w = std::ceil(ratio*(double)maxw);
      // Avoid duplicated: use "next one" if we duplicate a w
      const auto[_r, _w] = params->back();
      if(_w >= w){ w=_w+1; }
      // Write param
      params->emplace_back(std::tuple(i, w));
      // Break the loop if w=wmax
      if(w==maxw){ break; }
    }

    // Reverse param for LOOCV and fixup indexes
    std::reverse(params->begin(), params->end());
    for(size_t i=0; i<params->size(); ++i){ std::get<0>(params->at(i)) = i; }
    nb_params = params->size();
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Constructor / Destructor
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  DTW(tempo::DTS train, tempo::DTS test, double cfe) :
    tempo::classifier::nn1loocv::i_LOOCVDist(0),
    train(std::move(train)),
    test(std::move(test)),
    cfe(cfe),
    params(std::make_shared<std::vector<PType>>()) {
    generate_params();
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Interface implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  tempo::F distance_param(size_t train_idx1, size_t train_idx2, size_t param_idx, tempo::F bsf) override {
    const size_t w = std::get<1>(params->at(param_idx));
    return tempo::distance::univariate::dtw(train[train_idx1], train[train_idx2], cfe, w, bsf);
  }

  tempo::F distance_UB(size_t train_idx1, size_t train_idx2, size_t /* param_idx */) override {
    return tempo::distance::univariate::directa(train[train_idx1], train[train_idx2], cfe, tempo::utils::PINF);
  }

  tempo::F distance_test(size_t test_idx, size_t train_idx, tempo::F bsf) override {
    return tempo::distance::univariate::dtw(test[test_idx], train[train_idx], cfe, best_window, bsf);
  }

  void set_loocv_result(std::vector<size_t> bestp) override {
    // Report all the best in order - extract r and omega for json report
    // Note: sort on tuples, where first component is size_t
    std::sort(bestp.begin(), bestp.end());
    for (const auto& pi : bestp) {
      auto [i, omega] = params->at(pi);
      all_index.push_back(i);
      all_w.push_back(omega);
      std::cout << "parameter " << i << " penalty = " << omega << std::endl;
    }

    // Pick the smallest w, which is at the back of the array
    best_window = all_w.back();
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // JSON
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  nlohmann::json to_json() {
    nlohmann::json j;
    j["name"] = "DTW";
    j["best_indexes"] = tempo::utils::to_json(all_index);
    j["best_windows"] = tempo::utils::to_json(all_w);
    j["selected_window"] = best_window;
    j["cost_function_exponent"] = cfe;
    return j;
  }

};
