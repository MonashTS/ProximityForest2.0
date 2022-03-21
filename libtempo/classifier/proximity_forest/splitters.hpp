#pragma once

#include <libtempo/concepts.hpp>
#include <libtempo/tseries/dataset.hpp>
#include <libtempo/classifier/proximity_forest/pftree.hpp>

#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace libtempo::classifier::pf {

  template<Label L, typename State, typename PRNG>
  using SGenerators = std::vector<SplitterGenerator<L, State, PRNG>>;

  template<Label L, typename State, typename PRNG>
  [[nodiscard]] inline auto SplitterChooser(SGenerators<L, State, PRNG>&& generators) {
    return SplitterGenerator<L, State, PRNG>{
      .generate = [g = std::move(generators)](std::shared_ptr<State>& state,
                                              const IndexSet& is,
                                              const ByClassMap <L>& bcm,
                                              PRNG& prng) {
        const auto idx = std::uniform_int_distribution<size_t>(0, g.size() - 1)(prng);
        SplitterGenerator <L, State, PRNG>& gen = g[idx];
        return gen.generate(state, is, bcm, prng);
      }
    };
  }

  /// Splitter generator working with a dataset name and a distance.
  /// The generator will randomly choose the exemplar to compare against.
  template<Float F, Label L, State <L> S, typename PRNG, typename Dist>
  [[nodiscard]] inline auto NN1SplitterGenerator(std::string dsname, Dist distance) {
    return SplitterGenerator<L, S, PRNG>{.generate = [dsname, distance](std::shared_ptr<S>& state,
                                                                        const IndexSet& is,
                                                                        const ByClassMap <L>& bcm,
                                                                        PRNG& prng) {
      auto res = std::make_unique<Splitter<L, S, PRNG>>
      ();
      auto shared_state = std::make_shared<ByClassMap<L>>
      ();
      (Splitter<std::string, S, PRNG>{
        // Training: pick the exemplar
        .train=[](std::shared_ptr<S>& st, const IndexSet& is, const ByClassMap <L>& bcm, PRNG& prng) {

        },
        //
        .classify_train=[dsname, distance](std::shared_ptr<S>& state, size_t index, PRNG& prng) {
          const DTS <F, L>& dataset = get_dataset(*state);
          const auto& query = dataset[index];
          F bsf = utils::PINF<F>;
          size_t idx = index;

          return state->get_label(idx).value();
        },
        //


        .classify_test=[](std::shared_ptr<S>& state, size_t index, PRNG& prng) {
          return state->get_label(index).value();
        }
      });

      return res;
    }
    };
  }

}

