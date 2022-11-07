
/*
 Note: old (broken - need to be able to cache sample per transform in the state) code for ADTW.
 We now sample once and for all at the start of the algorithm.
 Faster, without accuracy loss (as per our experiments)

 We keep the code here if we one day want to put a sampling per node back in PF/TSChief

/// 1NN ADTW per node state
struct ADTWsAllGenState : public i_TreeState {
  // TODO: map : sample per transform!
  std::optional<F> sample{std::nullopt};

  // --- --- --- Constructor/Destructor

  // --- --- --- Implement interface

  std::unique_ptr<i_TreeState> forest_fork(size_t) const override {
    return std::make_unique<ADTWGenState>();
  }

  void forest_merge_in(std::unique_ptr<i_TreeState>&&) override {}

  /// When starting a new branch, reset the sample
  void start_branch(size_t) override {
    sample = std::nullopt;
  }

  void end_branch(size_t) override {}

};

struct ADTWsAllGen : public i_GenDist {

  static constexpr F omega_exponent = 5.0;

  TransformGetter get_transform;
  ExponentGetter get_fce;
  std::shared_ptr<i_GetState<ADTWsAllGenState>> get_adtw_state;
  std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data;

  ADTWsAllGen(
    TransformGetter gt,
    ExponentGetter get_cfe,
    std::shared_ptr<i_GetState<ADTWsAllGenState>> get_adtw_state,
    std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data
  ) :
    get_transform(std::move(gt)),
    get_fce(std::move(get_cfe)),
    get_adtw_state(std::move(get_adtw_state)),
    get_train_data(std::move(get_train_data)) {}

  std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) override {
    const std::string tn = get_transform(state);
    const F e = get_fce(state);

    // --- Sampling
    // Only sample if we haven't sample at this node yet. Cache the result if we compute it.
    // Automatically cleared when starting a new branch.
    std::optional<F>& sample = get_adtw_state->at(state).sample;
    if (!sample) {
      // Create subset
      size_t n = bcm.size();
      size_t SAMPLE_SIZE = std::min<size_t>(4000, n*(n - 1)/2);
      DTS train_subset(get_train_data->at(data).at(tn), "subset", bcm.to_IndexSet());
      // Sample random pairs
      tempo::utils::StddevWelford welford;
      std::uniform_int_distribution<> distrib(0, (int)train_subset.size() - 1);
      for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
        const auto& q = train_subset[distrib(state.prng)];
        const auto& s = train_subset[distrib(state.prng)];
        const F cost = distance::univariate::directa(q, s, e, utils::PINF);
        welford.update(cost);
      }
      // State updated here through mutable reference
      sample = {welford.get_mean()};
    }

    // --- Compute penalty
    const size_t i = std::uniform_int_distribution<size_t>(0, 100)(state.prng); // uniform, unbiased
    const F penalty = std::pow((F)i/100.0, omega_exponent)*sample.value();

    // Build return
    return std::make_unique<ADTW>(tn, e, penalty);
  }
};
 */
