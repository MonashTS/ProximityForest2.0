#pragma once

#include <libtempo/classifier/proximity_forest/isplitters.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>
#include <libtempo/distance/direct.hpp>

#include <random>
#include <utility>
#include <vector>
#include <iostream>

namespace libtempo::classifier::pf {

  namespace State {

    struct PRNG_mt64 {
      std::mt19937_64 prng;
      explicit PRNG_mt64(size_t seed) : prng(seed) {};
    };

    template<Float F, Label L>
    struct DatasetTS {
      using MAP_t = std::map<std::string, libtempo::DTS<F, L>>;
      using MAP_sptr_t = std::shared_ptr<MAP_t>;
      MAP_sptr_t dataset_ts;
      explicit DatasetTS(MAP_sptr_t map_sptr) : dataset_ts(std::move(map_sptr)) {}
    };

  }

  template<Label L, typename Strain, typename Stest>
  struct SG_chooser : public ISplitterGenerator<L, Strain, Stest> {
    using Result = typename ISplitterGenerator<L, Strain, Stest>::Result;

    using SGVec_t = std::vector<std::shared_ptr<ISplitterGenerator<L, Strain, Stest>>>;

    SGVec_t sgvec;
    size_t nb_candidates;

    SG_chooser(SGVec_t sgvev, size_t nb_candidates) :
      sgvec(std::move(sgvec)),
      nb_candidates(nb_candidates) {}

    /** Implementation fo the generate function
     * Randomly generate 'nb_candidates', evaluate them, keep the best (the lowest score is best)
     */
    std::unique_ptr<Result> generate(Strain& state, const ByClassMap<L>& bcm) const override {
      // Access the pseudo random number generator: state must inherit from PRNG_mt64
      auto& prng = static_cast<State::PRNG_mt64&>(state).prng;
      std::unique_ptr<Result> best_candidate{};
      double best_score = utils::PINF<double>;
      for (size_t i = 0; i<nb_candidates; ++i) {
        const auto idx = std::uniform_int_distribution<size_t>(0, sgvec.size() - 1)(prng);
        std::unique_ptr<Result> candidate = sgvec[idx]->generate(state, bcm);
        double score = weighted_gini_impurity(*candidate);
        if (score<best_score) {
          best_score = score;
          best_candidate = candidate;
        }
      }

      assert((bool)best_candidate); // Convert to bool: true if "has" a pointer, else false

      return best_candidate;
    }

  private:

    //// Compute the weighted (ratio of series per branch) gini impurity of a split.
    [[nodiscard]]
    static double weighted_gini_impurity(const Result& split) {
      double wgini{0};
      double total_size{0};
      for (const auto& variant : split.branch_splits) {
        switch (variant.index()) {
        case 0: {
          const ByClassMap<L>& bcm = std::get<0>(variant);
          double g = bcm.gini_impurity();
          // Weighted part: multiply gini score by the total number of item in this branch
          const double bcm_size = bcm.size();
          wgini += bcm_size*g;
          // Accumulate total size for final division
          total_size += bcm_size;
          break;
        }
        case 1: { /* Do nothing: contribute 0 to gini */ break; }
        default:libtempo::utils::should_not_happen();
        }
      }
      // Finish weighted computation by scaling to [0, 1]
      assert(total_size!=0);
      return wgini/total_size;
    }
  };

  /** 1NN Direct Alignment Test Time Splitter */
  template<Float F, Label L, typename Stest>
  struct Sp_1NN1DA : public ISplitter<L, Stest> {

    // Internal state
    std::map<L, size_t> labels_to_index;    // How to map label to branchs index
    IndexSet train_exemplar_is;             // IndexSet of the train exemplars (one per class)
    const DTS<F, L> *train_dataset;        // Reference to the train dataset
    std::string transformation_name;        // Which transformation to use

    /// Constructor
    Sp_1NN1DA(const DTS<F, L> *train_dataset, ByClassMap<L> bcm, std::string transformation_name) :
      labels_to_index(bcm.labels_to_index()),
      train_exemplar_is(bcm),
      train_dataset(train_dataset),
      transformation_name(std::move(transformation_name)) {}

    /// Classification
    size_t classify(Stest& state, size_t test_idx) override {
      auto& prng = static_cast<State::PRNG_mt64&>(state).prng;
      const DTS<F, L>& test_dataset = state.dataset_ts->at(transformation_name);
      const auto& query = test_dataset[test_idx];

      F bsf = utils::PINF<F>;
      std::vector<L> labels;
      // get list of labels for NN
      for (size_t exemplar_idx : train_exemplar_is) {
        const auto& exemplar = (*train_dataset)[exemplar_idx];
        auto dist = distance::univariate::directa(exemplar, query, bsf);
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
      //
      L predicted_label = utils::pick_one(labels, prng);
      return labels_to_index[predicted_label];
    }
  };

  /** 1NN Direct Alignment (DA) Splitter Generator
   *  Randomly Pick one exemplar per class.
   */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN1DA : public ISplitterGenerator<L, Strain, Stest> {
    using Result = typename ISplitterGenerator<L, Strain, Stest>::Result;

    /** Implementation fo the generate function */
    std::unique_ptr<Result> generate(Strain& state, const ByClassMap<L>& bcm) const override {
      std::string transformation_name = "default";
      // Access the pseudo random number generator: state must inherit from PRNG_mt64
      auto& prng = static_cast<State::PRNG_mt64&>(state).prng;
      ByClassMap<L> exemplars = bcm.template pick_one_by_class(prng);

      //const IndexSet& is = exemplars.get_IndexSet();
      const IndexSet& is = IndexSet(exemplars);
      // Access the dataset
      const DTS<F, L>& dataset = state.dataset_ts->at(transformation_name);
      // Build return
      auto labels_to_index = bcm.labels_to_index();
      std::vector<std::map<L, std::vector<size_t>>> v(bcm.nb_classes());
      // For each series in the incoming bcm (including selected exemplars - will eventually form pure leaves), 1NN
      //for (auto query_idx :bcm.get_IndexSet()) {
      for (auto query_idx :IndexSet(bcm)) {
        F bsf = utils::PINF<F>;
        std::vector<L> labels;
        const auto& query = dataset[query_idx];
        for (size_t exemplar_idx : is) {
          const auto& exemplar = dataset[exemplar_idx];
          auto dist = distance::univariate::directa<F, TSeries<F, L>>(exemplar, query, bsf);
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
        L predicted_label = utils::pick_one(labels, prng);
        // Update the branch: select the predicted label, but write the BCM with the real label
        size_t predicted_index = labels_to_index[predicted_label];
        L real_label = query.label().value();
        v[predicted_index][real_label].push_back(query_idx);
      }
      // Convert the vector of std::map in a vector of ByClassMap
      std::vector<ByClassMap<L>> v_bcm;
      for (auto&& m : v) { v_bcm.template emplace_back(std::move(m)); }
      // Build the test splitter
      auto r = std::unique_ptr<Result>(new Result{.branch_splits = std::move(v_bcm), .splitter=std::make_unique<Sp_1NN1DA<F, L, Stest>>(&dataset, exemplars, transformation_name)});
      return r;
    }

  };

}