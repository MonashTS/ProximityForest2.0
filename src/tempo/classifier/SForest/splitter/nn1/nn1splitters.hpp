#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include "tempo/classifier/SForest/stree.hpp"

namespace tempo::classifier::SForest::splitter::nn1 {

  // --- --- --- --- --- ---
  // State and Concepts

  /// Distance Splitter State components
  /// The forest train and test states must include a field "distance_splitter_state" of this type.
  struct NN1SplitterState {

    // --- --- --- --- --- ---
    // Per node cache. Must not be forked/merge

    /// IndexSet (i.e. per node) specific
    std::optional<IndexSet> dist_index_set{};

    /// Helper for the above
    const IndexSet& get_index_set(const ByClassMap& bcm) {
      if (!(bool)dist_index_set) { dist_index_set = std::make_optional<IndexSet>(bcm.to_IndexSet()); }
      return dist_index_set.value();
    }

    /// ADTW specific
    std::optional<double> ADTW_sampled_mean_da{};

    // --- --- --- --- --- ---
    // Constructor

    NN1SplitterState() = default;

    NN1SplitterState(NN1SplitterState&&) = default;

    NN1SplitterState& operator =(NN1SplitterState&&) = default;

    // --- --- --- --- --- ---
    // Deleted constructor

    NN1SplitterState(const NN1SplitterState&) = delete;


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // State Component Concept implementation

    /// Fork the state at the branch level.
    NN1SplitterState branch_fork(size_t /* bidx */) {
      return {};
    }

    /// Merge statistics, not cached data
    void branch_merge(NN1SplitterState&& /* other */) {}

  };
  static_assert(BaseState<NN1SplitterState>);

  /// Concept associated to NN1SplitterState: ensure the existance of the dfield "distance_splitter_state"
  template<typename S>
  concept HasNN1SplitterState = requires(S& s){
    s.distance_splitter_state;
  };

  /// NN1 Test Data concept: must give access to the train dataset
  template<typename D>
  concept NN1TestData = TestData<D> && requires(D& d, std::string tname){
    { d.get_train_dataset(tname) } -> std::convertible_to<DTS>;
  };

  // --- --- --- --- --- ---
  // Distance and Distance generator interfaces
  // Subclass each of these per distance

  /// Interface for the distance component: implement a child class
  struct Distance_i {

    // ---

    /// Distance function computing a similarity score between two series, the lower the more similar.
    /// 'bsf' ('Best so far') allows early abandoning and pruning in a NN1 classifier (upper bound on the whole process)
    virtual F eval(TSeries const& t1, TSeries const& t2, F bsf) = 0;

    // ---

    /// Name of the transformation to draw the data from
    virtual std::string get_transformation_name() = 0;

    // ---

    virtual ~Distance_i() = default;
  };

  /// NN1 Splitter Distance Generator: implement a child class for each distance for which we want a splitter
  template<typename TrainS, typename TrainD> requires HasNN1SplitterState<TrainS>
  struct NN1SplitterDistanceGen {

    /// Distance generator result
    /// Transmit back the state, the distance function and the transform name
    struct R {
      std::unique_ptr<TrainS> state;
      std::unique_ptr<Distance_i> distance;
    };

    virtual R generate(std::unique_ptr<TrainS> state, const TrainD& data, const ByClassMap& bcm) = 0;

    virtual ~NN1SplitterDistanceGen() = default;

  };


  // --- --- --- --- --- ---
  // Test time NN1 splitter
  // Hide it, avoid to pollute the namespace
  namespace {
    /// Test time NN1 Splitter for a distance implementing the NodeSplitter_i interface
    template<typename TestS, typename TestD>
    struct NN1Splitter : public NodeSplitter_i<TestS, TestD> {
      using R = typename NodeSplitter_i<TestS, TestD>::R;

      /// IndexSet of selected exemplar in the train
      IndexSet train_indexset;

      /// How to map label to index of branches
      std::map<EL, size_t> labels_to_branch_idx;

      /// Distance function
      std::unique_ptr<Distance_i> distance;

      NN1Splitter(IndexSet is, std::map<EL, size_t> labels_to_branch_idx, std::unique_ptr<Distance_i> dist) :
        train_indexset(std::move(is)),
        labels_to_branch_idx(std::move(labels_to_branch_idx)),
        distance(std::move(dist)) {}

      /// Interface implementation
      R get_branch_index(std::unique_ptr<TestS> state, const TestD& data, size_t test_index) override {
        // Distance info access
        std::string tname = distance->get_transformation_name();
        // Data access
        const DTS& test_dataset = data.get_test_dataset(tname);
        const DTS& train_dataset = data.get_train_dataset(tname);
        const TSeries& test_exemplar = test_dataset[test_index];

        // NN1 test loop
        F bsf = utils::PINF;
        std::set<EL> labels;
        for (size_t candidate_idx : train_indexset) {
          const auto& candidate = train_dataset[candidate_idx];
          F d = distance->eval(candidate, test_exemplar, bsf);
          if (d<bsf) {
            labels.clear();
            labels.insert(train_dataset.label(candidate_idx).value());
            bsf = d;
          } else if (bsf==d) { labels.insert(train_dataset.label(candidate_idx).value()); }
        }

        // Return the branch matching the predicted label
        EL predicted_label;
        std::sample(labels.begin(), labels.end(), &predicted_label, 1, state->prng);
        size_t idx = labels_to_branch_idx.at(predicted_label);

        return R{
          .state = std::move(state),
          .branch_index = idx
        };

      } // End of function get_branch_index
    };
  }

  // --- --- --- --- --- ---
  // Train time generator

  /// Train time NN1 Splitter Generator, to be built with a distance generator
  template<typename TrainS, typename TrainD, typename TestS, NN1TestData TestD> requires
  TreeState<TrainS>&&HasNN1SplitterState<TrainS>
  struct NN1SplitterGen : public NodeSplitterGen_i<TrainS, TrainD, TestS, TestD> {
    using R = typename NodeSplitterGen_i<TrainS, TrainD, TestS, TestD>::R;

    /// Distance generator object
    std::shared_ptr<NN1SplitterDistanceGen<TrainS, TrainD>> dgen;

    /// Construction with a distance generator
    explicit NN1SplitterGen(std::shared_ptr<NN1SplitterDistanceGen<TrainS, TrainD>> dgen) :
      dgen(std::move(dgen)) {}

    /// Generate a splitter based on the distance generator specifed at build time
    R generate(std::unique_ptr<TrainS> state0, const TrainD& data, const ByClassMap& bcm) override {

      // --- --- --- Generate a distance
      std::unique_ptr<TrainS> state1;
      std::unique_ptr<Distance_i> distance;
      std::string transform_name;
      {
        typename NN1SplitterDistanceGen<TrainS, TrainD>::R gen = dgen->generate(std::move(state0), data, bcm);
        state1 = std::move(gen.state);
        distance = std::move(gen.distance);
        transform_name = distance->get_transformation_name();
      }

      // --- --- --- Access State
      // Get/Compute the index set matching 'bcm'
      const IndexSet& all_indexset = state1->distance_splitter_state.get_index_set(bcm);

      // --- --- --- Access Train Data
      const DTS& train_dataset = data.get_train_dataset(transform_name);

      // --- --- --- Splitter training algorithm
      // Pick on exemplar per class using the pseudo random number generator from the state
      ByClassMap train_bcm = bcm.pick_one_by_class(state1->prng);
      IndexSet train_idxset = train_bcm.to_IndexSet();

      // Build return:
      //  Number of branches == number of classes
      //  We maintain mapping from labels to branch index
      //  We build the "by branch BCM" vector resulting from this splitter
      const std::map<EL, size_t>& label_to_branchIdx = bcm.labels_to_index();
      std::vector<ByClassMap::BCMvec_t> result_bcm_vec(bcm.nb_classes());

      // For each incoming series (including selected train exemplars - will eventually form pure leaves)
      // Do 1NN classification, managing ties
      for (auto query_idx : all_indexset) {
        const auto& query = train_dataset[query_idx];
        EL query_label = train_dataset.label(query_idx).value();
        // 1NN variables, use a set to manage ties
        F bsf = utils::PINF;
        std::set<EL> labels;
        //
        for (size_t candidate_idx : train_idxset) {
          const auto& candidate = train_dataset[candidate_idx];
          auto dist = distance->eval(candidate, query, bsf);
          if (dist<bsf) {
            labels.clear();
            labels.insert(train_dataset.label(candidate_idx).value());
            bsf = dist;
          } else if (bsf==dist) { labels.insert(train_dataset.label(candidate_idx).value()); }
        }
        // Break ties and choose the branch according to the predicted label
        tempo::EL predicted_label;
        std::sample(labels.begin(), labels.end(), &predicted_label, 1, state1->prng);
        size_t predicted_index = label_to_branchIdx.at(predicted_label);
        // The predicted label gives us the branch, but the BCM at the branch must contain the real label
        result_bcm_vec[predicted_index][query_label].push_back(query_idx);
      }

      // Convert the vector of ByClassMap::BCMvec_t in a vector of ByClassMap.
      // IMPORTANT: ensure that no empty BCM is generated
      // If we get an empty map, we have to add the  mapping (label for this index -> empty vector)
      // This ensures that no empty BCM is ever created. This is also why we iterate over the label: so we have them!
      std::vector<ByClassMap> v_bcm;
      for (EL label : bcm.classes()) {
        size_t idx = label_to_branchIdx.at(label);
        if (result_bcm_vec[idx].empty()) { result_bcm_vec[idx][label] = {}; }
        v_bcm.emplace_back(std::move(result_bcm_vec[idx]));
      }

      // Result
      return std::move(R{
        .state = std::move(state1),
        .splitter = std::make_unique<NN1Splitter<TestS, TestD>>(train_idxset, label_to_branchIdx, std::move(distance)),
        .branch_splits = std::move(v_bcm)
      });
    } // End of generate function

  };

}
