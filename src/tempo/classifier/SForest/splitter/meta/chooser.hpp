#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dataset.hpp>
#include "tempo/classifier/SForest/stree.hpp"

namespace tempo::classifier::SForest::splitter::meta {


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
  template<TreeState TrainS, typename TrainD, typename TestS, typename TestD>
  struct SplitterChooserGen : public NodeSplitterGen_i<TrainS, TrainD, TestS, TestD> {
    // Shorthands
    using NodeGen = NodeSplitterGen_i<TrainS, TrainD, TestS, TestD>;
    using SGVec_t = std::vector<std::shared_ptr<NodeGen>>;
    using R = typename NodeSplitterGen_i<TrainS, TrainD, TestS, TestD>::R;

    /// A vector of splitter generator
    SGVec_t sgvec;

    /// How many splitter candidate to generate
    size_t nb_candidates;

    SplitterChooserGen(SGVec_t&& sgvec, size_t nb_candidates) :
      sgvec(std::move(sgvec)),
      nb_candidates(nb_candidates) {}

    /// Implementation fo the generate function
    /// Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
    R generate(std::unique_ptr<TrainS> state, const TrainD& data, const ByClassMap& bcm) override {
      R best_result{};
      double best_score = utils::PINF;
      for (size_t i = 0; i<nb_candidates; ++i) {
        // Pick a splitter and call it
        R result = utils::pick_one(sgvec, state->prng)->generate(std::move(state), data, bcm);
        // Extract the state so we can transmit it
        state = std::move(result.state);
        double score = weighted_gini_impurity(result.branch_splits);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }
      // Put the state back into the result
      best_result.state = std::move(state);
      return best_result;
    }

  };

}