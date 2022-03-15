#pragma once

#include <libtempo/concepts.hpp>

#include <variant>

namespace libtempo::classifier::pf {

  namespace {

    /// Pure node: only contains a label. It is a leaf node.
    template<Label L>
    struct PureNode {
      L label;
    };

    /// Inner node.
    template<Float F, Label L>
    struct InnerNode {
      /// Da



    };

  }

  template<Float F, Label L>
  struct PFNode {
    /// Flag: is it a pure node or not?
    bool is_pure_node;

    /// Pure xor Inner node
    std::variant<PureNode<L>, std::variant<InnerNode<F,L>> node;

  };

}
