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
     * Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
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

  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData> requires has_prng<TrainState>
  struct SG_try_all : public IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData> {
    // Shorthands
    using INodeGen = IPF_NodeGenerator<L, TrainState, TrainData, TestState, TestData>;
    using Result = typename INodeGen::Result;
    using SGVec_t = std::vector<std::shared_ptr<INodeGen>>;

    SGVec_t sgvec;

    SG_try_all(SGVec_t&& sgvec) :
      sgvec(std::move(sgvec)) {}

    /** Implementation fo the generate function
     * Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
     */
    Result generate(TrainState& state, const TrainData& data, const BCMVec<L>& bcmvec) const override {
      Result best_result{};
      double best_score = utils::PINF<double>;
      for (size_t i = 0; i<sgvec.size(); ++i) {
        Result result = sgvec[i]->generate(state, bcmvec);
        double score = weighted_gini_impurity<L, TrainState, TrainData, TestState, TestData>(result);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }

      return best_result;
    }
  };

  /** Leaf generator, stopping at pure node */
  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct SGLeaf_PureNode : public IPF_LeafGenerator<L, TrainState, TrainData, TestState, TestData> {
    // Type shorthands
    using ILeafGen = IPF_LeafGenerator<L, TrainState, TrainData, TestState, TestData>;
    using Result = typename ILeafGen::Result;

    /// leaf splitter
    struct PureNode : public IPF_LeafSplitter<L, TestState, TestData> {

      double weight;
      std::vector<double> proba;

      PureNode(double weight, const std::map<L, size_t>& label_to_index, L label) :
        weight(weight),
        proba(label_to_index.size(), 0.0) {
        proba[label_to_index.at(label)] = 1.0;
      }

      std::tuple<double, std::vector<double>> predict_proba(TestState& /* state */ ,
                                                            const TestData& /* data */,
                                                            size_t /* test_index */) const override {
        return {weight, proba};
      }
    };

    /// Override interface ISplitterGenerator
    Result generate(TrainState& /* state */, const TrainData& data, const BCMVec<L>& bcmvec) const override {
      const auto& bcm = bcmvec.back();
      // Generate leaf on pure node
      if (bcm.nb_classes()==1) {
        const auto& header = data.get_header();
        std::string label = bcm.begin()->first;
        double weight = bcm.size();
        return {
          Result{
            ResLeaf<L, TrainState, TrainData, TestState, TestData>{
              .splitter = std::make_unique<PureNode>(weight, header.label_to_index(), label)}
          }
        };
      }
        // Else, return the empty option
      else { return {}; }
    }
  };

  /** Leaf generator, stopping if a node is pure, or at a given depth */
  template<Label L, typename TrainState, typename TrainData, typename TestState, typename TestData>
  struct SGLeaf_DepthNode : public IPF_LeafGenerator<L, TrainState, TrainData, TestState, TestData> {
    // Type shorthands
    using ILeafGen = IPF_LeafGenerator<L, TrainState, TrainData, TestState, TestData>;
    using Result = typename ILeafGen::Result;

    size_t depth_cutoff;

    explicit SGLeaf_DepthNode(size_t depth_cutoff) :
      depth_cutoff(depth_cutoff) {
    }

    /// leaf splitter
    struct DepthNode : public IPF_LeafSplitter<L, TestState, TestData> {

      double weight;
      std::vector<double> proba;

      DepthNode(double weight, std::vector<double>&& vec) :
        weight(weight),
        proba(std::move(vec)) {}

      std::tuple<double, std::vector<double>> predict_proba(TestState& /* state */ ,
                                                            const TestData& /* data */,
                                                            size_t /* test_index */) const override {
        return {weight, proba};
      }
    };

    /// Override interface ISplitterGenerator
    Result generate(TrainState& state, const TestData& data, const BCMVec<L>& bcmvec) const override {
      const auto& bcm = bcmvec.back();
      // Generate leaf on pure node
      if (bcm.nb_classes()==1) {
        const auto& header = state.get_header();
        std::string label = bcm.begin()->first;
        double weight = bcm.size();
        auto l_to_i = header.label_to_index();
        std::vector<double> proba(l_to_i.size(), 0.0);   // Allocate one per class
        size_t idx = l_to_i.at(label);
        proba[idx] = 1.0;
        return {
          Result{
            ResLeaf<L, TrainState, TrainData, TestState, TestData>{
              .splitter = std::make_unique<DepthNode>(weight, std::move(proba))
            }
          }
        };
      }
      // Stop at a given depth
      if (bcmvec.size()>=depth_cutoff) {
        const auto& header = state.get_header();
        double weight = bcm.size();
        auto l_to_i = header.label_to_index();
        std::vector<double> proba(l_to_i.size(), 0.0);   // Allocate one per class
        for (const auto& [label, vec] : bcm) {
          size_t idx = l_to_i.at(label);
          proba[idx] = ((double)vec.size())/weight;
        }
        return {
          Result{
            ResLeaf<L, TrainState, TrainData, TestState, TestData>{
              .splitter = std::make_unique<DepthNode>(weight, std::move(proba))
            }
          }};
      } else { return {}; } // Else, return the empty option
    }
  };

}