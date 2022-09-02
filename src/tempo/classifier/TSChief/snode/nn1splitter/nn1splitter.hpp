#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>

#include <tempo/classifier/TSChief/treedata.hpp>
#include <tempo/classifier/TSChief/treestate.hpp>
#include <tempo/classifier/TSChief/splitter_interface.hpp>

#include "nn1dist_interface.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

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
    i_GenNode::Result generate(TreeState& state, TreeData const& data, ByClassMap const& bcm) override;

  };

} // End of namespace tempo::classifier::TSChief::snode::nn1splitter
