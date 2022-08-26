#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dataset.hpp>
#include <tempo/classifier/sfdyn/stree.hpp>

namespace tempo::classifier::sf::node::meta {

  namespace {
    //// Compute the weighted (ratio of series per branch) gini impurity of a split.
    inline double weighted_gini_impurity(std::vector<ByClassMap> const& branch_split) {
      double wgini{0};
      double total_size{0};
      for (const auto& bcm : branch_split) {
        double g = bcm.gini_impurity();
        // Weighted part: multiply gini score by the total number of item in this branch
        auto bcm_size = (double)bcm.size();
        wgini += bcm_size*g;
        // Accumulate total size for final division
        total_size += bcm_size;
      }
      // Finish weighted computation by scaling to [0, 1]
      assert(total_size!=0);
      return wgini/total_size;
    }
  }

  /// Splitter generator chooser: choose between several other node generators
  struct SplitterChooserGen : public i_GenNode {

    // --- --- --- Fields
    /// A vector of splitter generator
    std::vector<std::shared_ptr<i_GenNode>> generators;

    /// How many splitter candidate to generate
    size_t nb_candidates;

    // --- --- --- Constructor/Destructor

    SplitterChooserGen(std::vector<std::shared_ptr<i_GenNode>>&& sgvec, size_t nb_candidates) :
      generators(std::move(sgvec)),
      nb_candidates(nb_candidates) {}

    // --- --- --- Method

    /// Implementation fo the generate function
    /// Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
    i_GenNode::Result generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) override {
      i_GenNode::Result best_result{};
      double best_score = utils::PINF;
      for (size_t i = 0; i<nb_candidates; ++i) {
        // Pick a splitter and call it
        i_GenNode::Result result = utils::pick_one(generators, state.prng)->generate(state, data, bcm);
        // Extract the state so we can transmit it
        double score = weighted_gini_impurity(result.branch_splits);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }
      // Put the state back into the result
      return best_result;
    }

  };

}