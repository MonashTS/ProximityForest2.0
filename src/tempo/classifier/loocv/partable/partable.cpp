#include "partable.hpp"

namespace tempo::classifier::loocv::partable {

  namespace {

    /// Our algorithm fills a 'LOOCV' table in parallel.
    /// Each cell of the table is locked by a mutex
    struct NNCell_mtx {
      /// Mutex over the cell
      std::mutex mutex{};
      /// index of the current NN
      size_t NNindex{};
      /// Distance to the current NN
      tempo::F NNdistance{};
    };

  }

  std::tuple<std::vector<size_t>, size_t> loocv_incparams(
    size_t                        nb_params,
    tempo::DatasetHeader const&   train_header,
    distance_fun_t                distance,
    upperbound_fun_t              upper_bound,
    size_t                        nb_threads
  ) {
    using namespace std;
    using namespace tempo;

    using NNC = NNCell_mtx;

    const size_t NBLINE = train_header.size();
    const size_t NBCOL = nb_params;
    const size_t NBCELL = NBLINE*NBCOL;

    if (NBLINE<2||nb_params==0) { throw invalid_argument("Can't do LOOCV with current arguments"); }

    // Always >=0 as per above
    const size_t LAST_P = nb_params - 1;

    // --- --- ---
    // Init

    // NNTable: One line per series, one columns per parameter
    // In a line, at each column, we will register the closest NN index and the associated distance
    // Init the table with "impossible" values for the index (out of bound) and +infinity for the distance
    vector<NNC> NNTable(NBCELL);
    for (size_t i = 0; i<NBCELL; ++i) {
      NNTable[i].NNindex = NBLINE;
      NNTable[i].NNdistance = utils::PINF;
    }

    // Ordered set of added series, sorted by their distance in NNTable[T, last]
    vector<size_t> intable;
    intable.reserve(NBLINE);


    // Helper function: thread-safe update a cell in the table (lock the mutex of the cell)
    auto update = [&](size_t table_idx, size_t nnindex, double nndist) {
      auto& nn = NNTable[table_idx];
      lock_guard lock(nn.mutex);
      if (nndist<nn.NNdistance) {
        nn.NNindex = nnindex;
        nn.NNdistance = nndist;
      } else if(nndist == nn.NNdistance){
        // TODO: manage ties

      }
    };

    // --- --- ---
    // Add the first pair
    {
      // Start with the upper bound
      {
        F UB = upper_bound(LAST_P, 0, 1);
        F d_last = distance(LAST_P, 0, 1, UB);
        update(0*NBCOL + NBCOL - 1, 1, d_last);
        update(1*NBCOL + NBCOL - 1, 0, d_last);
      }
      // Complete the table cutting with the UB (we are not going to EA anything here)
      for (size_t P = LAST_P; P>0;) {
        --P;
        // UB = distance from the following cell
        F UB = NNTable[0*NBCOL + P + 1].NNdistance;
        F d = distance(P, 0, 1, UB);
        update(0*NBCOL + P, 1, d);
        update(1*NBCOL + P, 0, d);
      }
      intable.push_back(0);
      intable.push_back(1);
    }

    // --- --- ---
    // Add the other series, in parallel

    utils::ParTasks ptask;
    // Some stats
    utils::duration_t filltable{0};
    utils::duration_t sort{0};

    for (size_t S = 2; S<NBLINE; ++S) {
      cout << "Doing " << S << "/" << NBLINE << "...";
      flush(cout);

      // Complete with other series 'Ti' already in the table
      // Create a task for each pair (S, Ti)
      for (size_t Ti : intable) {

        utils::ParTasks::task_t task = [&, Ti]() -> void {
          // Max bound: if above this, S and T cannot be each other NN
          const F dmax = max(NNTable[S*NBCOL + NBCOL - 1].NNdistance, NNTable[Ti*NBCOL + NBCOL - 1].NNdistance);
          // Start the process with the first parameters, and no lower bound
          size_t P = 0;
          F LB = 0;
          do {
            const F d_S = NNTable[S*NBCOL + P].NNdistance;
            const F d_Ti = NNTable[Ti*NBCOL + P].NNdistance;
            const F d_nn = max(d_S, d_Ti);
            if (LB<d_nn) {
              const F cutoff = min(dmax, upper_bound(P, S, Ti));
              const F di = distance(P, S, Ti, cutoff);
              update(S*NBCOL + P, Ti, di);
              update(Ti*NBCOL + P, S, di);
              if (di==utils::PINF) { P = NBCOL; } else { LB = di; }
            }
            P++;
          } while (P<NBCOL);
        };

        ptask.push_task(task);
      }

      auto filltable_start = utils::now();
      ptask.execute(nb_threads);
      auto filltable_duration = utils::now() - filltable_start;
      cout << "loop: " << utils::as_string(filltable_duration) << flush;

      // --- Put S in 'intable', maintaining approximate descending order on NNTable[S*NBCOL+NBCOL-1]
      auto sort_start = utils::now();
      {
        intable.push_back(S);
        for (size_t Tidx = intable.size() - 1; Tidx>=1; --Tidx) {
          const size_t Ti = intable[Tidx];
          const size_t Tiprev = intable[Tidx - 1];
          if (NNTable[Ti*NBCOL + NBCOL - 1].NNdistance>NNTable[Tiprev*NBCOL + NBCOL - 1].NNdistance) {
            swap(intable[Ti], intable[Tiprev]);
          }
        }
      }
      auto sort_duration = utils::now() - sort_start;
      cout << " sort: " << utils::as_string(sort_duration) << endl;

      filltable += filltable_duration;
      sort += sort_duration;
    }

    // --- NNTable is full: find the param with the fewest error
    vector<size_t> result;
    size_t bestError = numeric_limits<size_t>::max();
    {
      for (size_t pidx = 0; pidx<NBCOL; ++pidx) {
        size_t nError = 0;
        for (size_t Ti = 0; Ti<NBLINE; Ti++) {
          if (train_header.label(Ti).value()
            !=train_header.label(NNTable[Ti*NBCOL + pidx].NNindex).value()) { nError++; }
        }
        if (nError<bestError) {
          result.clear();
          result.push_back(pidx);
          bestError = nError;
        } else if (nError==bestError) { result.push_back(pidx); }
      }
    }

    cout << "Total fill table time = " << utils::as_string(filltable)
         << " Total sort time = " << utils::as_string(sort) << endl;

    return {result, NBLINE - bestError};
  }

} // End of namespace tempo::classifier::loocv_incparams::partable
