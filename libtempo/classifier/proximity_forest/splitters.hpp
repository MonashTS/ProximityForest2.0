#pragma once

#include <libtempo/classifier/proximity_forest/ipf.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>
#include <libtempo/distance/direct.hpp>
#include <libtempo/distance/dtw.hpp>

#include <random>
#include <utility>
#include <vector>
#include <iostream>
#include <functional>

namespace libtempo::classifier::pf {

  /** 1NN Test Time Splitter */
  template<Float F, Label L, typename Stest>
  requires has_prng<Stest> && TimeSeriesDataset<Stest, F, L>
  struct Sp_1NN : public IPF_NodeSplitter<L, Stest> {
    /// Distance function between two series, with a cutoff 'Best So Far' ('bsf') value
    using distance_t = std::function<F(const TSeries<F, L>& train_exemplar, const TSeries<F, L>& test_exemplar, F bsf)>;

    // Internal state
    DTS<F, L> train_dataset;                 /// Reference to the train dataset
    IndexSet train_indexset;                 /// IndexSet of the train exemplars (one per class)
    std::map<L, size_t> labels_to_index;     /// How to map label to index of branches
    std::string transformation_name;         /// Which transformation to use
    distance_t distance;                     /// Distance between two exemplars, accepting a cutoff

    Sp_1NN(DTS<F, L> TrainDataset,
           IndexSet TrainIndexset,
           std::map<L, size_t> LabelsToIndex,
           std::string TransformationName,
           distance_t Distance
    ) :
      train_dataset(std::move(TrainDataset)),
      train_indexset(std::move(TrainIndexset)),
      labels_to_index(std::move(LabelsToIndex)),
      transformation_name(std::move(TransformationName)),
      distance(std::move(Distance)) {}

    /// Splitter Classification
    size_t get_branch_index(Stest& state, size_t test_idx)
    override {
      auto& prng = state.prng;
      const auto& test_dataset_map = *state.dataset_shared_map;
      const auto& test_exemplar = test_dataset_map.at(transformation_name)[test_idx];
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
      L predicted_label = utils::pick_one(labels, *prng);
      // Return the branch matching the predicted label
      return labels_to_index[predicted_label];
    }
  };

  /** 1NN Splitter Generator - Randomly Pick one exemplar per class. */
  template<Float F, Label L, typename Strain, typename Stest>
  requires has_prng<Strain> && TimeSeriesDataset<Strain, F, L>
  struct SG_1NN : public IPF_NodeGenerator<L, Strain, Stest> {
    /// Use same distance type as the resulting test splitter
    using distance_t = typename Sp_1NN<F, L, Stest>::distance_t;
    /// Shorthand for the result type
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    /// Elastic distance between two series - must be already parameterized
    distance_t distance;
    /// Transformation name use to access the TimeSeriesDataset
    std::string transformation_name;

    SG_1NN(distance_t distance, std::string transformation_name) :
      distance(std::move(distance)), transformation_name(std::move(transformation_name)) {}

    /// Override generate function from interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      const ByClassMap<L>& bcm = bcmvec.back();
      // Pick on exemplar per class using the pseudo random number generator from the state
      auto& prng = state.prng;
      ByClassMap<L> train_bcm = bcm.template pick_one_by_class(*prng);
      const IndexSet& train_indexset = IndexSet(train_bcm);
      // Access the dataset
      const auto& train_dataset_map = *state.dataset_shared_map;
      const auto& train_dataset = train_dataset_map.at(transformation_name);
      // Build return
      auto labels_to_index = bcm.labels_to_index();
      std::vector<std::map<L, std::vector<size_t>>> result_bcmvec(bcm.nb_classes());
      // For each series in the incoming bcm (including selected exemplars - will eventually form pure leaves), 1NN
      for (auto query_idx : IndexSet(bcm)) {
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
        // Break ties
        L predicted_label = utils::pick_one(labels, *prng);
        // Update the branch: select the predicted label, but write the BCM with the real label
        size_t predicted_index = labels_to_index[predicted_label];
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
      return Result{ResNode<L, Stest>{
        .branch_splits = std::move(v_bcm),
        .splitter=std::make_unique<Sp_1NN<F, L, Stest>>(
          train_dataset, train_indexset, labels_to_index, transformation_name, distance
        )
      }};
    }
  };

  /** 1NN Direct Alignment Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_DA : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename Sp_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::string transformation_name;

    /// Exponent used in the cost function
    std::vector<double> exponents;

    SG_1NN_DA(std::string transformation_name, std::vector<double> exponents) :
      transformation_name(std::move(transformation_name)),
      exponents(std::move(exponents)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {

      double e = utils::pick_one(exponents, *state.prng);

      distance_t distance = [e](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::directa(t1, t2, distance::univariate::ade<F, TSeries<F, L>>(e), bsf);
      };

      SG_1NN<F, L, Strain, Stest> sg(distance, transformation_name);
      return sg.generate(state, bcmvec);
    }

  };

  /** 1NN DTW with full window Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_DTWFull : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename Sp_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::string transformation_name;

    /// Exponent used in the cost function
    std::vector<double> exponents;

    SG_1NN_DTWFull(std::string transformation_name, std::vector<double> exponents) :
      transformation_name(std::move(transformation_name)),
      exponents(std::move(exponents)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {

      double e = utils::pick_one(exponents, *state.prng);

      distance_t distance = [e](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::dtw(t1, t2, distance::univariate::ade<F, TSeries<F, L>>(e), bsf);
      };

      SG_1NN<F, L, Strain, Stest> sg(distance, transformation_name);
      return sg.generate(state, bcmvec);
    }

  };

  template<Label L, typename Strain, typename Stest>
  requires has_prng<Strain>
  struct SG_chooser : public IPF_NodeGenerator<L, Strain, Stest> {
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;

    using SGVec_t = std::vector<std::shared_ptr<IPF_NodeGenerator<L, Strain, Stest>>>;

    SGVec_t sgvec;
    size_t nb_candidates;

    SG_chooser(SGVec_t&& sgvec, size_t nb_candidates) :
      sgvec(std::move(sgvec)),
      nb_candidates(nb_candidates) {}

    /** Implementation fo the generate function
     * Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
     */
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      // Access the pseudo random number generator: state must inherit from PRNG_mt64
      auto& prng = state.prng;
      Result best_result{};
      double best_score = utils::PINF<double>;
      for (size_t i = 0; i<nb_candidates; ++i) {
        const auto idx = std::uniform_int_distribution<size_t>(0, sgvec.size() - 1)(*prng);
        Result result = sgvec[idx]->generate(state, bcmvec);
        double score = weighted_gini_impurity(result);
        if (score<best_score) {
          best_score = score;
          best_result = std::move(result);
        }
      }

      return best_result;
    }

  private:

    //// Compute the weighted (ratio of series per branch) gini impurity of a split.
    [[nodiscard]]
    static double weighted_gini_impurity(const Result& split) {
      double wgini{0};
      double total_size{0};
      for (const auto& bcm : split.branch_splits) {
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
  };


  /** Leaf generator, stopping at pure node */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SGLeaf_PureNode : public IPF_LeafGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_LeafGenerator<L, Strain, Stest>::Result;

    /// leaf splitter
    struct PureNode : public IPF_LeafSplitter<L, Stest> {
      std::vector<double> proba;

      PureNode(const std::map<L, size_t>& label_to_index, L label) : proba(label_to_index.size(), 0.0) {
        proba[label_to_index.at(label)] = 1.0;
      }

      std::vector<double> predict_proba(Stest& /* state */ , size_t /* test_index */) override {
        return proba;
      }
    };

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      const auto& bcm = bcmvec.back();
      // Generate leaf on pure node
      if (bcm.nb_classes()==1) {
        const auto& header = state.get_header();
        std::string label = bcm.begin()->first;
        return { Result{ResLeaf<L, Stest>{.splitter = std::make_unique<PureNode>(header.label_to_index(), label)}} };
      }
        // Else, return the empty option
      else { return {}; }
    }
  };


}