#pragma once

#include "ipf.hpp"
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>

#include <random>
#include <utility>
#include <vector>
#include <iostream>
#include <functional>

namespace libtempo::classifier::pf {

  //// Compute the weighted (ratio of series per branch) gini impurity of a split.
  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData>
  [[nodiscard]]
  static double weighted_gini_impurity(
    const typename IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData>::Result& result
  ) {
    double wgini{0};
    double total_size{0};
    for (const auto& bcm : result.branch_splits) {
      double g = bcm.gini_impurity();
      // Weighted part: multiply gini score by the total number of item in this branch
      const double bcm_size = bcm.size();
      wgini += bcm_size*g;
      // Accumulate total size for final division
      total_size += bcm_size;
    }
    // Finish weighted computation by scaling to [0, 1]
    assert(total_size!=0);
    return wgini/total_size;
  }

  /// Splitter generator chooser: choose between several other node generators
  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData> requires has_prng<TrainState>
  struct SG_chooser : public IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData> {
    // Shorthands
    using INodeGen = IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData>;
    using Result = typename INodeGen::Result;
    using SGVec_t = std::vector<std::shared_ptr<INodeGen>>;

    SGVec_t sgvec;
    size_t nb_candidates;

    SG_chooser(SGVec_t&& sgvec, size_t nb_candidates) :
      sgvec(std::move(sgvec)),
      nb_candidates(nb_candidates) {}

    /** Implementation fo the generate function
     *  Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
     */
    Result generate(TrainState& state, const TrainData& data, const BCMVec<L>& bcmvec) const override {
      Result best_result{};
      double best_score = utils::PINF<double>;
      for (size_t i = 0; i<nb_candidates; ++i) {
        Result result = utils::pick_one(sgvec, *state.prng)->generate(state, data, bcmvec);
        double score = weighted_gini_impurity<L, TrainState, TrainData, TestState, TestData>(result);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }

      return best_result;
    }
  };

  /// Pure leaf generator: stop when only one class reaches the node
  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct SGLeaf_PureNode : public IPF_LeafGenerator<L, TrainState, TrainData, TestState, TestData> {
    // Type shorthands
    using ILeafGen = IPF_LeafGenerator<L, TrainState, TrainData, TestState, TestData>;
    using Result = typename ILeafGen::Result;

    /// leaf splitter
    struct PureNode : public IPF_LeafSplitter<L, TestState, TestData> {

      arma::Col<size_t> cardinality;

      PureNode(size_t size, const std::map<L, size_t>& label_to_index, L label)
        : cardinality(label_to_index.size(), arma::fill::zeros) {
        cardinality[label_to_index.at(label)] = size_t(size);
      }

      arma::Col<size_t> predict_cardinality(TestState& /* state */ ,
                                            const TestData& /* data */,
                                            size_t /* test_index */) const override { return cardinality; }

    };

    /// Override interface ISplitterGenerator
    Result generate(TrainState& /* state */, const TrainData& data, const BCMVec<L>& bcmvec) const override {
      const auto& bcm = bcmvec.back();
      // Generate leaf on pure node
      if (bcm.nb_classes()==1) {
        const auto& header = data.get_header();
        std::string label = bcm.begin()->first;
        size_t size = bcm.size();
        return {
          Result{
            ResLeaf<L, TrainState, TrainData, TestState, TestData>{
              .splitter = std::make_unique<PureNode>(size, header.label_to_index(), label)}
          }
        };
      }
        // Else, return the empty option
      else { return {}; }
    }
  };

}