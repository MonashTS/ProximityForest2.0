#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>

#include <map>
#include <string>
#include <variant>

namespace libtempo::classifier::pf {

  /** For PF purpose, a datastore is a map(std::string, DTS<F,L>) */
  template<Float F, Label L>
  using Store = std::map<std::string, DTS<F,L>>;


  /** Function extracting a Store from a an arbitrary piece of data */
  template<typename F, typename Fl, typename La, typename D>
  concept GetStore = requires(const D& data, F fun){
    { fun(data) }->std::same_as<const Store<Fl, La>&>;
  };


  /** Splitter interface
   *  A Splitter is analogous to a classifier than can be trained on a dataset.
   */
  template<Float F, Label L, typename Data, GetStore<F,L,Data> FGetStore, typename PRNG>
  struct Splitter {
    std::function<std::vector<L>(const Data& d, FGetStore access, PRNG& prng)> train;
    std::function<std::vector<L>(const Data& d, FGetStore access, PRNG& prng)> test;
  };

  /** Interface: generate a new splitter, given a vector of transforms and a source of randomness.
   *  Note: Method is const as it should be thread-shareable. */
  template<typename FloatType, typename LabelType, typename PRNG>
  struct SplitterGenerator {
    using DS = Dataset<FloatType, LabelType>;
    using Splitter_ptr = std::unique_ptr<Splitter<FloatType, LabelType>>;

    virtual Splitter_ptr get_splitter(
      const DS& ds, const IndexSet& is,
      const ByClassMap<LabelType>& exemplars, PRNG& prng) = 0;

    virtual ~SplitterGenerator() = default;
  };





  template<Float F, Label L>
  struct PFNode {
    /// Dataset type
    using DS = libtempo::DTS<F, L>;

    /// Pure node: only contains a label. It is a leaf node.
    struct Leaf { L label; };

    /// Inner node.
    struct Node {


      /// Mapping between a class (label) to a branch (subnode)
      using Branches = std::unordered_map<L, std::unique_ptr<PFNode>>;

    };

    /// Flag: is it a pure node or not?
    bool is_pure_node;

    /// Pure xor Inner node
    std::variant<Leaf, Node> node;

    /** Build a tree */
    template<typename PRNG>
    static std::unique_ptr<PFNode> make_tree(
      const DS &ds, const IndexSet &is, const ByClassMap<L> bcm, size_t nbcandidates, SplitterGenerator<F, L, PRNG> sg, PRNG &prng) {

    }

  };

}
