#pragma once
#include <libtempo/classifier/proximity_forest/ipf.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/concepts.hpp>

#include <libtempo/distance/direct.hpp>
#include <libtempo/distance/adtw.hpp>
#include <libtempo/distance/dtw.hpp>
#include <libtempo/distance/cdtw.hpp>
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

  namespace internal {

    /** 1NN Test Time Splitter */
    template<Float F, Label L, typename Stest> requires has_prng<Stest>&&TimeSeriesDataset<Stest, F, L>
    struct TestSplitter_1NN : public IPF_NodeSplitter<L, Stest> {

      /// Distance function between two series, with a cutoff 'Best So Far' ('bsf') value
      using distance_t =
      std::function<F(const TSeries<F, L>& train_exemplar, const TSeries<F, L>& test_exemplar, F bsf)>;

      // Internal state
      DTS<F, L> train_dataset;                 /// Reference to the train dataset
      IndexSet train_indexset;                 /// IndexSet of the train exemplars (one per class)
      std::map<L, size_t> labels_to_index;     /// How to map label to index of branches
      std::string transformation_name;         /// Which transformation to use
      distance_t distance;                     /// Distance between two exemplars, accepting a cutoff

      TestSplitter_1NN(DTS<F, L> TrainDataset,
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
      size_t get_branch_index(Stest& state, size_t test_idx) const override {
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
        return labels_to_index.at(predicted_label);
      }
    };

    /** 1NN Splitter Generator - Randomly Pick one exemplar per class. */
    template<Float F, Label L, typename Strain, typename Stest> requires has_prng<Strain>
      &&TimeSeriesDataset<Strain, F, L>
    struct TrainSplitter_1NN : public IPF_NodeGenerator<L, Strain, Stest> {

      /// Use same distance type as the resulting test splitter
      using distance_t = typename TestSplitter_1NN<F, L, Stest>::distance_t;

      /// Callback function
      using callback_t = std::function<void(Strain&)>;

      /// Shorthand for the result type
      using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;

      /// Elastic distance between two series - must be already parameterized
      distance_t distance;

      /// Transformation name use to access the TimeSeriesDataset
      std::string transformation_name;

      /// Callback for the result
      callback_t cb;

      TrainSplitter_1NN(distance_t distance, std::string transformation_name, callback_t cb = [](Strain&) {}) :
        distance(std::move(distance)), transformation_name(std::move(transformation_name)), cb(cb) {}

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
        return Result{ResNode<L, Strain, Stest>{
          .branch_splits = std::move(v_bcm),
          .splitter=std::make_unique<TestSplitter_1NN<F, L, Stest>>(
            train_dataset, train_indexset, labels_to_index, transformation_name, distance
          ),
          .callback = cb
        }};
      }
    };

  } // End of namespace internal


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Elastic distance splitters generators
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /** 1NN Direct Alignment */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_DA : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Exponent used in the cost function
    std::shared_ptr<std::vector<double>> exponents;

    SG_1NN_DA(std::shared_ptr<std::vector<std::string>> transformation_names,
              std::shared_ptr<std::vector<double>> exponents) :
      transformation_names(std::move(transformation_names)),
      exponents(std::move(exponents)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {

      std::string tname = utils::pick_one(*transformation_names, *state.prng);
      double e = utils::pick_one(*exponents, *state.prng);

      distance_t distance = [e](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::directa(t1, t2, distance::univariate::ade<F, TSeries<F, L >>(e), bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["da"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }

  };

  /** 1NN DTW Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_DTW : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Exponent used in the cost function
    std::shared_ptr<std::vector<double>> exponents;

    SG_1NN_DTW(std::shared_ptr<std::vector<std::string>> transformation_names,
               std::shared_ptr<std::vector<double>> exponents) :
      transformation_names(std::move(transformation_names)),
      exponents(std::move(exponents)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {

      // Compute the window
      const size_t win_top = (state.get_header().length_max() + 1)/4;
      const auto w = std::uniform_int_distribution<size_t>(0, win_top)(*state.prng);

      std::string tname = utils::pick_one(*transformation_names, *state.prng);
      double e = utils::pick_one(*exponents, *state.prng);

      distance_t distance = [e, w](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::cdtw(t1, t2, w, distance::univariate::ade<F, TSeries<F, L >>(e), bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["dtw"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }
  };

  /** 1NN DTW with full window Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_DTWFull : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Exponent used in the cost function
    std::shared_ptr<std::vector<double>> exponents;

    SG_1NN_DTWFull(std::shared_ptr<std::vector<std::string>> transformation_names,
                   std::shared_ptr<std::vector<double>> exponents) :
      transformation_names(std::move(transformation_names)),
      exponents(std::move(exponents)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {

      std::string tname = utils::pick_one(*transformation_names, *state.prng);
      double e = utils::pick_one(*exponents, *state.prng);

      distance_t distance = [e](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::dtw(t1, t2, distance::univariate::ade<F, TSeries<F, L >>(e), bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["dtwf"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }
  };

  /** 1NN ADTW Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_ADTW : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Exponent used in the cost function
    std::shared_ptr<std::vector<double>> exponents;

    /// Sampling size when assessing the penalty
    size_t penalty_sampling_size;

    /// Herrmann Schedule - divisor
    size_t hs_div;

    /// Herrmann Schedule - exponent
    double hs_exp;

    SG_1NN_ADTW(std::shared_ptr<std::vector<std::string>> transformation_names,
                std::shared_ptr<std::vector<double>> exponents,
                size_t penalty_sampling_size,
                size_t hs_div,
                double hs_exp) :
      transformation_names(std::move(transformation_names)),
      exponents(std::move(exponents)),
      penalty_sampling_size(penalty_sampling_size),
      hs_div(hs_div),
      hs_exp(hs_exp) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      std::string tname = utils::pick_one(*transformation_names, *state.prng);
      double e = utils::pick_one(*exponents, *state.prng);

      auto dist = distance::univariate::ade<F, TSeries<F, L >>(e);

      // --- --- --- Sampling
      const auto& train_dataset_map = *state.dataset_shared_map;
      const auto& train_dataset = train_dataset_map.at(tname);
      utils::StddevWelford welford;
      {
        for (size_t i = 0; i<penalty_sampling_size; ++i) {
          const auto& q = utils::pick_one(train_dataset.data(), *state.prng);
          const auto& s = utils::pick_one(train_dataset.data(), *state.prng);
          const auto cost = distance::directa<F>(q, s, dist);
          welford.update(cost);
        }
      }
      const double sampled_mean_dist = welford.get_mean();

      // Get a x in [0, div]
      const double x = std::uniform_int_distribution<int>(0, hs_div + 1)(*state.prng);
      const double r = std::pow(x/(double)hs_div, hs_exp);
      const double omega = r*sampled_mean_dist;

      distance_t distance = [e, omega, dist](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::adtw(t1, t2, omega, dist, bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["adtw"] += 1; };

      auto sg = internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb);
      return sg.generate(state, bcmvec);
    }
  };

  /** 1NN WDTW Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_WDTW : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Exponent used in the cost function
    std::shared_ptr<std::vector<double>> exponents;

    SG_1NN_WDTW(std::shared_ptr<std::vector<std::string>> transformation_names,
                std::shared_ptr<std::vector<double>> exponents) :
      transformation_names(std::move(transformation_names)),
      exponents(std::move(exponents)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {

      // Compute the weight vector
      const F g = std::uniform_real_distribution<F>(0, 1)(*state.prng);
      auto weights = std::make_shared<std::vector<F>>(distance::generate_weights(g, state.get_header().length_max()));

      std::string tname = utils::pick_one(*transformation_names, *state.prng);
      double e = utils::pick_one(*exponents, *state.prng);

      distance_t distance = [e, weights](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::wdtw(t1, t2, *weights, distance::univariate::ade<F, TSeries<F, L >>(e), bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["wdtw"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }
  };

  /** 1NN ERP Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_ERP : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Exponent used in the cost function
    std::shared_ptr<std::vector<double>> exponents;

    SG_1NN_ERP(std::shared_ptr<std::vector<std::string>> transformation_names,
               std::shared_ptr<std::vector<double>> exponents) :
      transformation_names(std::move(transformation_names)),
      exponents(std::move(exponents)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      std::string tname = utils::pick_one(*transformation_names, *state.prng);
      double e = utils::pick_one(*exponents, *state.prng);

      // Compute the window
      const size_t win_top = (state.get_header().length_max() + 1)/4;
      const auto w = std::uniform_int_distribution<size_t>(0, win_top)(*state.prng);

      // Compute the gap value using the standard deviation of the data reaching this node
      const auto& train_dataset_map = *state.dataset_shared_map;
      const auto& train_dataset = train_dataset_map.at(tname);
      auto stddev_ = stddev(train_dataset, IndexSet(bcmvec.back()));
      const double gv = std::uniform_real_distribution<double>(0.2*stddev_, stddev_)(*state.prng);

      distance_t distance = [e, w, gv](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::erp(t1, t2, w, gv
                             , distance::univariate::adegv<F, TSeries<F, L>>(e)
                             , distance::univariate::ade<F, TSeries<F, L >>(e)
                             , bsf
        );
      };

      auto cb = [](Strain& strain) { strain.selected_distances["erp"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }
  };

  /** 1NN LCSS Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_LCSS : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    explicit SG_1NN_LCSS(std::shared_ptr<std::vector<std::string>> transformation_names) :
      transformation_names(std::move(transformation_names)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      std::string tname = utils::pick_one(*transformation_names, *state.prng);

      // Compute the window
      const size_t win_top = (state.get_header().length_max() + 1)/4;
      const auto w = std::uniform_int_distribution<size_t>(0, win_top)(*state.prng);

      // Compute the epsilon value using the standard deviation of the data reaching this node
      const auto& train_dataset_map = *state.dataset_shared_map;
      const auto& train_dataset = train_dataset_map.at(tname);
      auto stddev_ = stddev(train_dataset, IndexSet(bcmvec.back()));
      const double epsilon = std::uniform_real_distribution<double>(0.2*stddev_, stddev_)(*state.prng);

      distance_t distance = [w, epsilon](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::univariate::lcss(t1, t2, w, epsilon, bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["lcss"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }
  };

  /** 1NN MSM Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_MSM : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Set of possible costs
    std::shared_ptr<std::vector<double>> costs;

    SG_1NN_MSM(
      std::shared_ptr<std::vector<std::string>> transformation_names,
      std::shared_ptr<std::vector<double>> costs
    ) :
      transformation_names(std::move(transformation_names)),
      costs(std::move(costs)) {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      std::string tname = utils::pick_one(*transformation_names, *state.prng);

      // Compute the cost
      auto c = utils::pick_one(*costs, *state.prng);

      distance_t distance = [c](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::univariate::msm(t1, t2, c, bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["msm"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }
  };

  /** 1NN TWE Splitter Generator */
  template<Float F, Label L, typename Strain, typename Stest>
  struct SG_1NN_TWE : public IPF_NodeGenerator<L, Strain, Stest> {
    // Type shorthands
    using Result = typename IPF_NodeGenerator<L, Strain, Stest>::Result;
    using distance_t = typename internal::TestSplitter_1NN<F, L, Stest>::distance_t;

    /// Transformation name
    std::shared_ptr<std::vector<std::string>> transformation_names;

    /// Set of possible nu
    std::shared_ptr<std::vector<double>> nus;

    /// Set of possible lambda
    std::shared_ptr<std::vector<double>> lambdas;

    SG_1NN_TWE(
      std::shared_ptr<std::vector<std::string>> transformation_names,
      std::shared_ptr<std::vector<double>> nus,
      std::shared_ptr<std::vector<double>> lambdas
    ) :
      transformation_names(std::move(transformation_names)),
      nus(std::move(nus)),
      lambdas(std::move(lambdas))
      {}

    /// Override interface ISplitterGenerator
    Result generate(Strain& state, const std::vector<ByClassMap<L>>& bcmvec) const override {
      std::string tname = utils::pick_one(*transformation_names, *state.prng);

      // Pick nu and lambda
      auto nu = utils::pick_one(*nus, *state.prng);
      auto lambda = utils::pick_one(*lambdas, *state.prng);

      distance_t distance = [nu, lambda](const TSeries<F, L>& t1, const TSeries<F, L>& t2, double bsf) {
        return distance::univariate::twe<F, TSeries<F,L>>(t1, t2, nu, lambda, bsf);
      };

      auto cb = [](Strain& strain) { strain.selected_distances["twe"] += 1; };

      return internal::TrainSplitter_1NN<F, L, Strain, Stest>(distance, tname, cb).generate(state, bcmvec);
    }
  };

}