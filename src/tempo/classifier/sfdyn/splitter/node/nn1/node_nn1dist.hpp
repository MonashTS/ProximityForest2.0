#pragma once

#include <string>

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/sfdyn/stree.hpp>
#include <utility>

namespace tempo::classifier::sf::node::nn1dist {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Time series distance interface and base implementation

  /// Interface for the distance component
  struct i_Dist {

    // --- --- --- Destructor/Constructor

    virtual ~i_Dist() = default;

    // --- --- --- Method

    /// Distance function computing a similarity score between two series, the lower the more similar.
    /// 'bsf' ('Best so far') allows early abandoning and pruning in a NN1 classifier (upper bound on the whole process)
    virtual F eval(TSeries const& t1, TSeries const& t2, F bsf) = 0;

    /// Name of the transformation to draw the data from
    virtual std::string get_transformation_name() = 0;

    virtual std::string get_distance_name() = 0;
  };

  struct BaseDist : public i_Dist {

    // --- --- --- Destructor/Constructor

    explicit BaseDist(std::string str) : transformation_name(std::move(str)) {}

    ~BaseDist() override = default;

    // --- --- --- Method

    /// Store the name of the transform
    std::string transformation_name;

    /// Name of the transformation to draw the data from
    std::string get_transformation_name() override { return transformation_name; }

  };

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // NN1 Time Series Distance Splitter

  struct SplitterNN1 : public i_SplitterNode {

    // --- --- --- Types

    /// IndexSet of selected exemplar in the train
    IndexSet train_indexset;

    /// How to map label to index of branches
    std::map<EL, size_t> labels_to_branch_idx;

    /// Distance function
    std::unique_ptr<i_Dist> distance;

    /// Train Data access
    std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data;

    /// Test Data access
    std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_test_data;

    // --- --- --- Constructors/Destructors

    SplitterNN1(
      IndexSet is,
      std::map<EL, size_t> labels_to_branch_idx,
      std::unique_ptr<i_Dist> dist,
      std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_train_data,
      std::shared_ptr<i_GetData<std::map<std::string, DTS>>> get_test_data
    ) :
      train_indexset(std::move(is)),
      labels_to_branch_idx(std::move(labels_to_branch_idx)),
      distance(std::move(dist)),
      get_train_data(std::move(get_train_data)),
      get_test_data(std::move(get_test_data)) {}

    // --- --- --- Methods

    size_t get_branch_index(TreeState& tstate, TreeData const& tdata, size_t index) override {
      // Distance info access
      std::string tname = distance->get_transformation_name();

      // Data access
      const DTS& train_dataset = get_train_data->at(tdata).at(tname);
      const DTS& test_dataset = get_test_data->at(tdata).at(tname);
      const TSeries& test_exemplar = test_dataset[index];

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
      assert(!labels.empty());

      // Return the branch matching the predicted label
      EL predicted_label;
      std::sample(labels.begin(), labels.end(), &predicted_label, 1, tstate.prng);
      return labels_to_branch_idx.at(predicted_label);

    } // End of function get_branch_index
  };

} // End of namespace tempo::classifier::sf::node::nn1dist
