#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/classifier/proximity_forest/pftree.hpp>

#include <functional>
#include <random>
#include <vector>

namespace libtempo::classifier::pf {

  /** Create a splitter using a generator uniformly chosen at random between in a list of generators */
  template<Label L, typename State, typename PRNG>
  static auto SplitterChooser(std::vector<SplitterGenerator<L, State, PRNG>>&& generators) {

    return SplitterGenerator<L, State, PRNG>{
      .generate = [g = std::move(generators)](
        std::shared_ptr<State>& state,
        const IndexSet& is,
        const ByClassMap<L>& bcm,
        PRNG& prng
      ) {
        const auto idx = std::uniform_int_distribution<size_t>(0, g.size()-1)(prng);
        SplitterGenerator<L, State, PRNG>& gen = g[idx];
        return gen.generate(state, is, bcm, prng);
      }
    };

  }

}
