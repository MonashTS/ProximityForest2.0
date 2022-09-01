#pragma once

#include <string>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include <tempo/classifier/TSChief/treedata.hpp>
#include <tempo/classifier/TSChief/treestate.hpp>
#include "node_nn1dist.hpp"

namespace tempo::classifier::TSChief::snode::nn1dist {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Time series distance generator interface

  struct i_GenDist {

    virtual ~i_GenDist() = default;

    virtual std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) = 0;

  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Specific state for NN1 snode generator

  struct GenSplitterNN1_State : public i_TreeState {

    // --- --- --- Fields

    /// Per node IndexSet cache
    std::optional<IndexSet> cache_index_set{};

    // --- --- --- Constructors/Constructors

    GenSplitterNN1_State() = default;

    GenSplitterNN1_State(GenSplitterNN1_State&&) = default;

    GenSplitterNN1_State& operator =(GenSplitterNN1_State&&) = default;

    GenSplitterNN1_State(const GenSplitterNN1_State&) = delete;

    ~GenSplitterNN1_State() override = default;

    // --- --- --- Methods

    /// Helper for the index set
    const IndexSet& get_index_set(const ByClassMap& bcm) {
      if (!cache_index_set) { cache_index_set = std::make_optional<IndexSet>(bcm.to_IndexSet()); }
      return cache_index_set.value();
    }

    std::unique_ptr<i_TreeState> forest_fork(size_t /* tree_idx */) const override {
      return std::unique_ptr<i_TreeState>(new GenSplitterNN1_State());
    }

    void forest_merge_in(std::unique_ptr<i_TreeState>&& /* other */ ) override { /* nothing */ }

    void start_branch(size_t /* branch_idx */) override { cache_index_set = {}; }

    void end_branch(size_t /* branch_idx */) override { /* nothing */ }
  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // NN1 Time Series Distance Splitter Generator

  struct GenSplitterNN1 : public i_GenNode {

    // --- --- --- Types

    // --- --- --- Fields

    /// Distance generator object
    std::shared_ptr<i_GenDist> distance_generator;

    /// Train State access
    std::shared_ptr<i_GetState<GenSplitterNN1_State>> get_train_state;

    /// Train Data access
    std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data;

    /// Test Data access
    std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_test_data;

    // --- --- --- Constructors/Destructors

    /// Construction with a distance generator
    GenSplitterNN1(
      std::shared_ptr<i_GenDist> distance_generator,
      std::shared_ptr<i_GetState<GenSplitterNN1_State>> get_train_state,
      std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data,
      std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_test_data
    ) :
      distance_generator(std::move(distance_generator)),
      get_train_state(std::move(get_train_state)),
      get_train_data(std::move(get_train_data)),
      get_test_data(std::move(get_test_data)) {}

    // --- --- --- Methods

    /// Generate a snode based on the distance generator specifed at build time
    i_GenNode::Result generate(TreeState& state, TreeData const& data, ByClassMap const& bcm) override {

      // --- --- --- Generate a distance
      auto distance = distance_generator->generate(state, data, bcm);
      std::string transform_name = distance->get_transformation_name();

      // --- --- --- Access State
      // Get/Compute the index set matching 'bcm'
      const IndexSet& all_indexset = get_train_state->at(state).get_index_set(bcm);

      // --- --- --- Data access
      const DTS& train_dataset = get_train_data->at(data).at(transform_name);

      // --- --- --- Splitter training algorithm
      // Pick on exemplar per class using the pseudo random number generator from the state
      ByClassMap train_bcm = bcm.pick_one_by_class(state.prng);
      IndexSet train_idxset = train_bcm.to_IndexSet();

      // Build return:
      //  Number of branches == number of classes
      //  We maintain mapping from labels to branch index
      //  We build the "by branch BCM" vector resulting from this snode
      const std::map<EL, size_t>& label_to_branchIdx = bcm.labels_to_index();
      std::vector<ByClassMap::BCMvec_t> result_bcm_vec(bcm.nb_classes());

      // For each incoming series (including selected train exemplars - will eventually form pure leaves)
      // Do 1NN classification, managing ties
      for (auto query_idx : all_indexset) {
        const auto& query = train_dataset[query_idx];
        EL query_label = train_dataset.label(query_idx).value();

        // 1NN variables, use a set to manage ties
        // Start with same class: better chance to have a tight cutoff
        std::set<EL> labels = {query_label};
        const size_t first_idx = train_bcm[query_label][0];
        F bsf = distance->eval(train_dataset[first_idx], query, utils::PINF);

        // Continue with other classes
        for (size_t candidate_idx : train_idxset) {
          if (candidate_idx!=first_idx) {
            const auto& candidate = train_dataset[candidate_idx];
            auto dist = distance->eval(candidate, query, bsf);
            if (dist<bsf) {
              labels.clear();
              labels.insert(train_dataset.label(candidate_idx).value());
              bsf = dist;
            } else if (bsf==dist) { labels.insert(train_dataset.label(candidate_idx).value()); }
          }
        }

        // Break ties and choose the branch according to the predicted label
        tempo::EL predicted_label;
        std::sample(labels.begin(), labels.end(), &predicted_label, 1, state.prng);
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

      return i_GenNode::Result{
        .splitter = std::make_unique<SplitterNN1>(train_idxset, label_to_branchIdx, std::move(distance),
                                                  get_train_data, get_test_data),
        .branch_splits = std::move(v_bcm)
      };
    } // End of generate function
  };

} // End of namespace tempo::classifier::TSChief::snode::nn1dist
