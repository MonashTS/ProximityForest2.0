#pragma once
#include "../ipf.hpp"
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>

#include <libtempo/distance/direct.hpp>
#include <libtempo/distance/adtw.hpp>
#include <libtempo/distance/dtw.hpp>
#include <libtempo/distance/wdtw.hpp>
#include <libtempo/distance/erp.hpp>
#include <libtempo/distance/lcss.hpp>
#include <libtempo/distance/msm.hpp>
#include <libtempo/distance/twe.hpp>

#include <random>
#include <utility>
#include <vector>
#include <functional>

namespace libtempo::classifier::pf {

  /// Distance Splitter concept. Act as a distance function while containing information about itself.
  template<typename Dist, typename F, typename L>
  concept DistanceSplitter =
  Float<F>&&Label<L>&&requires(const Dist& distance, const TSeries<F, L>& t1, const TSeries<F, L>& t2, F bsf){
    { distance(t1, t2, bsf) } -> std::convertible_to<F>;
    { distance.transformation_name() } -> std::convertible_to<std::string>;
  };

  /// Distance Generator concept: produces a DistanceSplitter (see above).
  template<typename DistGen, typename F, typename L, typename TrainState, typename TrainData>
  concept DistanceGenerator = requires(
    const DistGen& mk_distance,
    TrainState& state,
    const TrainData& data,
    const BCMVec<L>& bcmvec
  ){
    { mk_distance(state, data, bcmvec) } -> DistanceSplitter<F, L>;
  };

  namespace internal {
    /// Test Time 1NN Splitter
    template<Float F, Label L, typename TestState, typename TestData, DistanceSplitter<F, L> Distance>
    struct Splitter_1NN : public IPF_NodeSplitter<L, TestState, TestData> {

      IndexSet train_indexset;                 /// IndexSet of selected exemplar in the train
      std::map<L, size_t> labels_to_index;     /// How to map label to index of branches
      Distance distance;

      /// Mixin construction: must provide basic components
      Splitter_1NN(IndexSet is, std::map<L, size_t> m, Distance distance) :
        train_indexset(std::move(is)),
        labels_to_index(std::move(m)),
        distance(std::move(distance)) {}

      /// Interface override (Splitter Classification)
      size_t get_branch_index(TestState& state, const TestData& data, size_t test_index) const override {
        // State access
        auto& prng = state.prng;
        // Data access
        const DTS<F, L>& test_dataset = data.get_test_dataset(distance.transformation_name());
        const TSeries<F, L>& test_exemplar = test_dataset[test_index];
        const DTS<F, L>& train_dataset = data.get_train_dataset(distance.transformation_name());
        // NN1 test loop
        F bsf = utils::PINF<F>;
        std::vector<L> labels;
        for (size_t train_idx : train_indexset) {
          const auto& train_exemplar = train_dataset[train_idx];
          F d = distance(train_exemplar, test_exemplar, bsf);
          if (d<bsf) {
            labels = {train_exemplar.label().value()};
            bsf = d;
          } else if (bsf==d) {
            const auto& l = train_exemplar.label().value();
            if (std::none_of(labels.begin(), labels.end(), [l](const auto& v) { return v==l; })) {
              labels.emplace_back(l);
            }
          }
        }
        // Return the branch matching the predicted label
        L predicted_label = utils::pick_one(labels, *prng);
        return labels_to_index.at(predicted_label);
      } // End of function get_branch_index

    }; // End of struct Mixin_1NN_TestTimeSplitter
  } // End of namespace internal

  /// Train Time 1NN Splitter Generator
  template<
    Float F, Label L, typename TrainState, typename TrainData, typename TestState, typename TestData,
    DistanceGenerator<F, L, TrainState, TrainData> DistanceGenerator
  >
  struct SG_1NN : public IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData> {

    /// Shorthand for the Result type
    using Result = typename IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData>::Result;

    DistanceGenerator mk_distance;

    explicit SG_1NN(DistanceGenerator mk_distance) :
      mk_distance(std::move(mk_distance)) {}

    /// Override generate function from interface ISplitterGenerator
    Result generate(TrainState& state, const TrainData& data, const std::vector<ByClassMap<L>>& bcmvec)
    const override {

      // --- --- --- Generate splitter using Generator
      auto distance = mk_distance(state, data, bcmvec);
      using Distance = decltype(distance);

      // --- --- --- Access BCM
      const ByClassMap<L>& bcm = bcmvec.back();

      // --- --- --- Access State
      auto& prng = state.prng;
      // Get/Compute the index set matching 'bcm'
      const IndexSet& all_indexset = state.distance_splitter_state.get_index_set(bcm);

      // --- --- --- Access Train Data
      const DTS<F, L>& train_dataset = data.get_train_dataset(distance.transformation_name());

      // --- --- --- Splitter training algorithm
      // Pick on exemplar per class using the pseudo random number generator from the state
      ByClassMap<L> train_bcm = bcm.template pick_one_by_class(*prng);
      IndexSet train_indexset(train_bcm);

      // Build return
      auto labels_to_index = bcm.labels_to_index();
      std::vector<std::map<L, std::vector<size_t>>> result_bcmvec(bcm.nb_classes());

      // For each series in the incoming bcm (including selected exemplars - will eventually form pure leaves), 1NN
      for (auto query_idx : all_indexset) {
        F bsf = utils::PINF<F>;
        std::vector<L> labels;
        const auto& query = train_dataset[query_idx];
        for (size_t exemplar_idx : train_indexset) {
          const auto& exemplar = train_dataset[exemplar_idx];
          auto dist = distance(exemplar, query, bsf);
          if (dist<bsf) {
            labels.clear();
            labels.template emplace_back(exemplar.label().value());
            bsf = dist;
          } else if (bsf==dist) {
            const auto& l = exemplar.label().value();
            if (std::none_of(labels.begin(), labels.end(), [l](const auto& v) { return v==l; })) {
              labels.emplace_back(l);
            }
          }
        }
        // Break ties and update the branch: select the predicted label, but write the BCM with the real label
        L predicted_label = utils::pick_one(labels, *prng);
        size_t predicted_index = labels_to_index.at(predicted_label);
        L real_label = query.label().value();
        result_bcmvec[predicted_index][real_label].push_back(query_idx);
      }
      // Convert the vector of std::map in a vector of ByClassMap.
      // IMPORTANT: ensure that no empty BCM is generated
      // If we get an empty map, we have to add the  mapping (label for this index -> empty vector)
      // This ensures that no empty BCM is ever created. This is also why we iterate over the label: so we have them!
      std::vector<ByClassMap<L>> v_bcm;
      for (const auto& label : bcm.classes()) {
        size_t idx = labels_to_index[label];
        if (result_bcmvec[idx].empty()) { result_bcmvec[idx][label] = {}; }
        v_bcm.emplace_back(std::move(result_bcmvec[idx]));
      }
      // Build the splitter
      return Result{ResNode<L, TestState, TestData>{
        .branch_splits = std::move(v_bcm),
        .splitter = std::make_unique<internal::Splitter_1NN<F, L, TestState, TestData, Distance>>(
          train_indexset, labels_to_index, distance
        )
      }};
    }
  }; // End of struct SplitterGenerator_1NN


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Generators used by most elastic distance splitter generators
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Generate a transform name suitable for elastic distances
  template<typename TrainState>
  using TransformGetter = std::function<std::string(TrainState& train_state)>;

  /// Generate an exponent e used in some elastic distances' cost function cost(a,b)=|a-b|^e
  template<typename TrainState>
  using ExponentGetter = std::function<double(TrainState& train_state)>;

  /// Generate a warping window
  template<typename TrainState, typename TrainData>
  using WindowGetter = std::function<size_t(TrainState& train_state, const TrainData& train_data)>;


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Function used by all distances
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Distances need to know the transform on which they operate. We abstract this in this struct.
  struct TName {
    std::string tname;

    explicit TName(std::string tname) : tname(std::move(tname)) {}

    [[nodiscard]]
    const std::string& transformation_name() const { return tname; }
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Splitters and Splitter Generators
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// 1NN Direct Alignment, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_DA : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      TransformGetter<TrainState> get_transform;
      ExponentGetter<TrainState> get_exponent;

      Generator(TransformGetter<TrainState> gt, ExponentGetter<TrainState> ge) :
        get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_DA operator ()(TrainState& state, const TrainData&, const BCMVec<L>&) const {
        std::string tn = get_transform(state);
        double e = get_exponent(state);
        return Splitter_1NN_DA(tn, e);
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// ADP cost function exponent
    double exponent;

    /// Constructor
    Splitter_1NN_DA(std::string tname, double exponent) :
      TName(std::move(tname)),
      exponent(exponent) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      return distance::directa(t1, t2, distance::univariate::ade<F, TSeries<F, L >>(exponent), bsf);
    }

  };

  /// 1NN DTW Full Window, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_DTWFull : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      TransformGetter<TrainState> get_transform;
      ExponentGetter<TrainState> get_exponent;

      Generator(TransformGetter<TrainState> gt, ExponentGetter<TrainState> ge) :
        get_transform(std::move(gt)), get_exponent(std::move(ge)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_DTWFull operator ()(TrainState& state, const TrainData&, const BCMVec<L>&) const {
        std::string tn = get_transform(state);
        double e = get_exponent(state);
        return Splitter_1NN_DTWFull(tn, e);
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// ADP cost function exponent
    double exponent;

    /// Constructor
    Splitter_1NN_DTWFull(std::string tname, double exponent) :
      TName(std::move(tname)),
      exponent(exponent) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      return distance::dtw(t1, t2, distance::univariate::ade<F, TSeries<F, L >>(exponent), utils::NO_WINDOW, bsf);
    }

  };

  /// 1NN DTW with parametric window, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_DTW : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      TransformGetter<TrainState> get_transform;
      ExponentGetter<TrainState> get_exponent;
      WindowGetter<TrainState, TrainData> get_window;

      Generator(TransformGetter<TrainState> gt, ExponentGetter<TrainState> ge, WindowGetter<TrainState, TrainData> gw) :
        get_transform(std::move(gt)),
        get_exponent(std::move(ge)),
        get_window(std::move(gw)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_DTW operator ()(TrainState& state, const TrainData& data, const BCMVec<L>&) const {
        std::string tn = get_transform(state);
        double e = get_exponent(state);
        size_t w = get_window(state, data);

        return Splitter_1NN_DTW(tn, e, w);
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// ADP cost function exponent
    double exponent;

    /// Warping window
    size_t window;

    /// Constructor
    Splitter_1NN_DTW(std::string tname, double exponent, size_t window) :
      TName(std::move(tname)),
      exponent(exponent),
      window(window) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      return distance::dtw(t1, t2, distance::univariate::ade<F, TSeries<F, L >>(exponent), window, bsf);
    }
  };

  /// 1NN WDTW with parametric weights, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_WDTW : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      TransformGetter<TrainState> get_transform;
      ExponentGetter<TrainState> get_exponent;

      Generator(TransformGetter<TrainState> gt, ExponentGetter<TrainState> ge) :
        get_transform(std::move(gt)),
        get_exponent(std::move(ge)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_WDTW operator ()(TrainState& state, const TrainData& data, const BCMVec<L>&) const {
        std::string tn = get_transform(state);
        double e = get_exponent(state);
        //
        const F g = std::uniform_real_distribution<F>(0, 1)(*state.prng);
        auto w = std::vector<F>(distance::generate_weights(g, data.get_header().length_max()));
        //
        return Splitter_1NN_WDTW(tn, e, std::move(w));
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// ADP cost function exponent
    double exponent;

    /// Precomputed weights - must accommodate longest series
    std::vector<F> weights;

    /// Constructor
    Splitter_1NN_WDTW(std::string tname, double exponent, std::vector<F>&& weights) :
      TName(std::move(tname)),
      exponent(exponent),
      weights(std::move(weights)) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      return distance::wdtw(t1, t2, distance::univariate::ade<F, TSeries<F, L >>(exponent), weights, bsf);
    }
  };

  /// 1NN ERP with parametric window and gap value, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_ERP : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      TransformGetter<TrainState> get_transform;
      ExponentGetter<TrainState> get_exponent;
      WindowGetter<TrainState, TrainData> get_window;

      Generator(TransformGetter<TrainState> gt, ExponentGetter<TrainState> ge, WindowGetter<TrainState, TrainData> gw) :
        get_transform(std::move(gt)),
        get_exponent(std::move(ge)),
        get_window(std::move(gw)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_ERP operator ()(TrainState& state, const TrainData& data, const BCMVec<L>& bcmvec) const {
        std::string tn = get_transform(state);
        double e = get_exponent(state);
        size_t w = get_window(state, data);
        // gap value
        const auto& train_dataset = data.get_train_dataset(tn);
        auto stddev_ = stddev(train_dataset, IndexSet(bcmvec.back()));
        const F g = std::uniform_real_distribution<F>(0.2*stddev_, stddev_)(*state.prng);
        //
        return Splitter_1NN_ERP(tn, e, w, g);
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// ADP cost function exponent
    double exponent;

    /// PWarping Window
    size_t window;

    /// Gap value
    F gv;

    /// Constructor
    Splitter_1NN_ERP(std::string tname, double exponent, size_t w, F gv) :
      TName(std::move(tname)),
      exponent(exponent),
      window(w),
      gv(gv) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      auto gvdist = distance::univariate::adegv<F, TSeries<F, L >>(exponent);
      auto dist = distance::univariate::ade<F, TSeries<F, L >>(exponent);
      return distance::erp(t1, t2, gvdist, dist, window, gv, bsf);
    }
  };

  /// 1NN LCSS with parametric window and gap value, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_LCSS : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      TransformGetter<TrainState> get_transform;
      WindowGetter<TrainState, TrainData> get_window;

      Generator(TransformGetter<TrainState> gt, WindowGetter<TrainState, TrainData> gw) :
        get_transform(std::move(gt)),
        get_window(std::move(gw)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_LCSS operator ()(TrainState& state, const TrainData& data, const BCMVec<L>& bcmvec) const {
        std::string tn = get_transform(state);
        size_t w = get_window(state, data);
        // Epsilon value
        const auto& train_dataset = data.get_train_dataset(tn);
        auto stddev_ = stddev(train_dataset, IndexSet(bcmvec.back()));
        const F e = std::uniform_real_distribution<F>(0.2*stddev_, stddev_)(*state.prng);
        //
        return Splitter_1NN_LCSS(tn, w, e);
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Warping Window
    size_t window;

    /// Gap value
    F epsilon;

    /// Constructor
    Splitter_1NN_LCSS(std::string tname, size_t w, F epsilon) :
      TName(std::move(tname)),
      window(w),
      epsilon(epsilon) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      return distance::lcss(t1, t2, distance::univariate::ad1<F, TSeries<F, L>>, window, epsilon, bsf);
    }
  };

  /// 1NN MSM with parametric window and gap value, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_MSM : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      using CostGetter = std::function<F(TrainState& state)>;

      TransformGetter<TrainState> get_transform;
      CostGetter get_cost;

      Generator(TransformGetter<TrainState> gt, CostGetter gc) :
        get_transform(std::move(gt)),
        get_cost(std::move(gc)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_MSM operator ()(TrainState& state, const TrainData&, const BCMVec<L>&) const {
        std::string tn = get_transform(state);
        F c = get_cost(state);
        return Splitter_1NN_MSM(tn, c);
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Cost
    F cost;

    /// Constructor
    Splitter_1NN_MSM(std::string tname, F cost) :
      TName(std::move(tname)),
      cost(cost) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      return distance::univariate::msm(t1, t2, cost, bsf);
    }
  };

  /// 1NN TWE with parametric window and gap value, Splitter + Generator as nested class
  template<Float F, Label L>
  struct Splitter_1NN_TWE : public TName {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    template<typename TrainState, typename TrainData>
    struct Generator {

      using Getter = std::function<F(TrainState& state)>;

      TransformGetter<TrainState> get_transform;
      Getter get_nu;
      Getter get_lambda;

      Generator(TransformGetter<TrainState> gt, Getter g_nu, Getter g_lambda) :
        get_transform(std::move(gt)),
        get_nu(std::move(g_nu)),
        get_lambda(std::move(g_lambda)) {}

      /// Generator requirement: create a distance
      Splitter_1NN_TWE operator ()(TrainState& state, const TrainData&, const BCMVec<L>&) const {
        std::string tn = get_transform(state);
        F n = get_nu(state);
        F l = get_lambda(state);
        return Splitter_1NN_TWE(tn, n, l);
      }

    }; // End of struct Generator

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// TWE Nu
    F nu;

    /// TWE Lamnda
    F lambda;

    /// Constructor
    Splitter_1NN_TWE(std::string tname, F nu, F lambda) :
      TName(std::move(tname)), nu(nu), lambda(lambda) {}

    /// Concept Requirement: compute the distance between two series
    [[nodiscard]]
    F operator ()(const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) const {
      return distance::univariate::twe<F, TSeries<F, L>>(t1, t2, nu, lambda, bsf);
    }
  };

  /// Distance Splitter State components
  /// The forest train and test states must include a field "distance_splitter_state" of this type.
  template<Label L>
  struct DistanceSplitterState : public pf::IStateComp<L, DistanceSplitterState<L>> {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Statistics
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    bool do_instrumentation;

    /// Map of selected distances.
    /// A new empty map is created when branching, and later merged with other maps.
    std::map<std::string, size_t> selected_distances;

    /// Update distance usage statistics
    void update(const std::string& distname) {
      if (do_instrumentation) {
        selected_distances[distname] += 1;
      }
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Cache
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// IndexSet specific: do not recompute the IndexSet every time
    std::optional<IndexSet> dist_index_set;

    /// Helper for the above
    [[nodiscard]] const IndexSet& get_index_set(const ByClassMap<L>& bcm) {
      if (!(bool)dist_index_set) {
        dist_index_set = std::make_optional<IndexSet>(bcm);
      }
      return dist_index_set.value();
    }

    /// ADTW specific
    std::optional<double> ADTW_sampled_mean_da;


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructors
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    DistanceSplitterState() = delete;

    DistanceSplitterState(const DistanceSplitterState&) = delete;

    explicit DistanceSplitterState(bool do_instrumentation) : do_instrumentation(do_instrumentation) {}


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Overrides
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Branch the state. Per-node states do not contain data (empty options, map...)
    DistanceSplitterState<L> fork(size_t /* bidx */) override {
      return DistanceSplitterState<L>(do_instrumentation);
    }

    /// Merge statistics, not cached data
    void merge(const DistanceSplitterState<L>& other) override {
      for (const auto& [n, c] : other.selected_distances) {
        selected_distances[n] += c;
      }
    }

    /// On leaf: nothing
    void on_leaf(const BCMVec<L>& /* bcmvec */ ) override {}
  };

  namespace internal {

    // /** 1NN Test Time Splitter */
    // template<Float F, Label L, typename Stest> requires distance_splitter_acceptable<Stest, F, L>
    // struct TestSplitter_1NN : public IPF_NodeSplitter<L, Stest> {

    //   /// Distance function between two series, with a cutoff 'Best So Far' ('bsf') value
    //   using distance_t =
    //     std::function<F(const TSeries<F, L>& train_exemplar, const TSeries<F, L>& test_exemplar, F bsf)>;

    //   // Internal state
    //   DTS<F, L> train_dataset;                 /// Reference to the train dataset
    //   IndexSet train_indexset;                 /// IndexSet of the train exemplars (one per class)
    //   std::map<L, size_t> labels_to_index;     /// How to map label to index of branches
    //   std::string transformation_name;         /// Which transformation to use
    //   distance_t distance;                     /// Distance between two exemplars, accepting a cutoff

    //   TestSplitter_1NN(DTS<F, L> TrainDataset,
    //                    IndexSet TrainIndexset,
    //                    std::map<L, size_t> LabelsToIndex,
    //                    std::string TransformationName,
    //                    distance_t Distance
    //   ) :
    //     train_dataset(std::move(TrainDataset)),
    //     train_indexset(std::move(TrainIndexset)),
    //     labels_to_index(std::move(LabelsToIndex)),
    //     transformation_name(std::move(TransformationName)),
    //     distance(std::move(Distance)) {}

    //   /// Splitter Classification
    //   size_t get_branch_index(Stest& state, size_t test_idx) const override {
    //     auto& prng = state.prng;
    //     const auto& test_dataset_map = *state.dataset_shared_map;
    //     const auto& test_exemplar = test_dataset_map.at(transformation_name)[test_idx];
    //     // NN1 test loop
    //     F bsf = utils::PINF<F>;
    //     std::vector<L> labels;
    //     for (size_t train_idx : train_indexset) {
    //       const auto& train_exemplar = train_dataset[train_idx];
    //       F d = distance(train_exemplar, test_exemplar, bsf);
    //       if (d<bsf) {
    //         labels = {train_exemplar.label().value()};
    //         bsf = d;
    //       } else if (bsf==d) {
    //         const auto& l = train_exemplar.label().value();
    //         if (std::none_of(labels.begin(), labels.end(), [l](const auto& v) { return v==l; })) {
    //           labels.emplace_back(l);
    //         }
    //       }
    //     }
    //     L predicted_label = utils::pick_one(labels, *prng);
    //     // Return the branch matching the predicted label
    //     return labels_to_index.at(predicted_label);
    //   }
    // };

    // /** 1NN Splitter Generator - Randomly Pick one exemplar per class. */
    // template<Float F, Label L, typename Strain, typename Stest>
    // struct TrainSplitter_1NN : public IPF_NodeGenerator<L, Strain, Stest> {

    //   /// Use same distance type as the resulting test splitter
    //   using distance_t = typename TestSplitter_1NN<F, L, Stest>::distance_t;

    //   /// Callback function
    //   using callback_t = std::function<void(Strain&)>;

    //   /// Shorthand for the result type
    //   using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;

    //   /// Elastic distance between two series - must be already parameterized
    //   distance_t distance;

    //   /// Transformation name use to access the TimeSeriesDataset
    //   std::string transformation_name;

    //   /// Callback for the result
    //   callback_t cb;

    //   TrainSplitter_1NN(distance_t distance, std::string transformation_name, callback_t cb = [](Strain&) {}) :
    //     distance(std::move(distance)), transformation_name(std::move(transformation_name)), cb(cb) {}

    //   /// Override generate function from interface ISplitterGenerator
    //   Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
    //     const ByClassMap<L>& bcm = bcmvec.back();
    //     // Pick on exemplar per class using the pseudo random number generator from the state
    //     auto& prng = state.prng;
    //     ByClassMap<L> train_bcm = bcm.template pick_one_by_class(*prng);
    //     // Check the state for the index set
    //     const IndexSet& train_indexset = state.distance_splitter_state.get_index_set(train_bcm);
    //     // Access the dataset
    //     const auto& train_dataset_map = *state.dataset_shared_map;
    //     const auto& train_dataset = train_dataset_map.at(transformation_name);
    //     // Build return
    //     auto labels_to_index = bcm.labels_to_index();
    //     std::vector<std::map<L, std::vector<size_t>>> result_bcmvec(bcm.nb_classes());
    //     // For each series in the incoming bcm (including selected exemplars - will eventually form pure leaves), 1NN
    //     for (auto query_idx : IndexSet(bcm)) {
    //       F bsf = utils::PINF<F>;
    //       std::vector<L> labels;
    //       const auto& query = train_dataset[query_idx];
    //       for (size_t exemplar_idx : train_indexset) {
    //         const auto& exemplar = train_dataset[exemplar_idx];
    //         auto dist = distance(exemplar, query, bsf);
    //         if (dist<bsf) {
    //           labels.clear();
    //           labels.template emplace_back(exemplar.label().value());
    //           bsf = dist;
    //         } else if (bsf==dist) {
    //           const auto& l = exemplar.label().value();
    //           if (std::none_of(labels.begin(), labels.end(), [l](const auto& v) { return v==l; })) {
    //             labels.emplace_back(l);
    //           }
    //         }
    //       }
    //       // Break ties
    //       L predicted_label = utils::pick_one(labels, *prng);
    //       // Update the branch: select the predicted label, but write the BCM with the real label
    //       size_t predicted_index = labels_to_index.at(predicted_label);
    //       L real_label = query.label().value();
    //       result_bcmvec[predicted_index][real_label].push_back(query_idx);
    //     }
    //     // Convert the vector of std::map in a vector of ByClassMap.
    //     // IMPORTANT: ensure that no empty BCM is generated
    //     // If we get an empty map, we have to add the  mapping (label for this index -> empty vector)
    //     // This ensures that no empty BCM is ever created. This is also why we iterate over the label: so we have them!
    //     std::vector<ByClassMap<L>> v_bcm;
    //     for (const auto& label : bcm.classes()) {
    //       size_t idx = labels_to_index[label];
    //       if (result_bcmvec[idx].empty()) { result_bcmvec[idx][label] = {}; }
    //       v_bcm.emplace_back(std::move(result_bcmvec[idx]));
    //     }
    //     // Build the splitter
    //     return Result{ResNode<L, Strain, Stest>{
    //       .branch_splits = std::move(v_bcm),
    //       .splitter=std::make_unique<TestSplitter_1NN<F, L, Stest>>(
    //       train_dataset, train_indexset, labels_to_index, transformation_name, distance
    //       ),
    //       .callback = cb
    //     }};
    //   }
    // };

  } // End of namespace internal

}


//   /** 1NN ADTW Splitter Generator */
//   template<Float F, Label L, typename Strain, typename Stest>
//   struct SG_1NN_ADTW : public IPF_NodeGenerator<L, Strain, Stest> {
//     // Type shorthands
//     using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
//     using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

//     /// Transformation name
//     std::shared_ptr<std::vector<std::string>> transformation_names;

//     /// Exponent used in the cost function
//     std::shared_ptr<std::vector<double>> exponents;

//     /// Mean direct alignment sampling size
//     size_t mean_da_sampling_size;

//     /// Herrmann Schedule - sampling size
//     size_t hs_sampling_size;

//     /// Herrmann Schedule - exponent
//     double hs_exp;

//     SG_1NN_ADTW(std::shared_ptr<std::vector<std::string>> transformation_names,
//                 std::shared_ptr<std::vector<double>> exponents,
//                 size_t mean_da_sampling_size,
//                 size_t hs_sampling_size,
//                 double hs_exp) :
//       transformation_names(std::move(transformation_names)),
//       exponents(std::move(exponents)),
//       mean_da_sampling_size(mean_da_sampling_size),
//       hs_sampling_size(hs_sampling_size),
//       hs_exp(hs_exp) {}

//     /// Override interface ISplitterGenerator
//     Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
//       std::string tname = utils::pick_one(*transformation_names, *state.prng);
//       double e = utils::pick_one(*exponents, *state.prng);

//       auto dist = distance::univariate::ade<F, TSeries<F, L >>(e);

//       // --- --- --- Sampling
//       // lazy-shared, i.e. do not resample if already done at this node
//       if(!(bool)state.distance_splitter_state.ADTW_sampled_mean_da){
//         const auto& train_dataset_map = *state.dataset_shared_map;
//         const auto& train_dataset = train_dataset_map.at(tname);
//         utils::StddevWelford welford;
//         {
//           for (size_t i = 0; i<mean_da_sampling_size; ++i) {
//             const auto& q = utils::pick_one(train_dataset.data(), *state.prng);
//             const auto& s = utils::pick_one(train_dataset.data(), *state.prng);
//             const auto cost = distance::directa<F>(q, s, dist);
//             welford.update(cost);
//           }
//         }
//         state.distance_splitter_state.ADTW_sampled_mean_da = std::make_optional<double>(welford.get_mean());
//       }
//       double sampled_mean_da = state.distance_splitter_state.ADTW_sampled_mean_da.value();


//       // Get a x in [0, sampling_size]
//       const double x = std::uniform_int_distribution<int>(0, hs_sampling_size + 1)(*state.prng);
//       const double r = std::pow(x/(double)hs_sampling_size, hs_exp);
//       const double omega = r*sampled_mean_da;

//       distance_t distance = [e, omega, dist](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
//         return distance::adtw(t1, t2, dist, omega, bsf);
//       };

//       auto cb = [tname](Strain& strain) { strain.distance_splitter_state.update("adtw_" + tname); };

//       auto sg = internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb);
//       return sg.generate(state, bcmvec);
//     }
//   };
