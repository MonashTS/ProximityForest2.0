#pragma once

#include <ostream>
#include <iostream>

namespace tempo {


  struct ProgressMonitor {

    size_t total;

    explicit ProgressMonitor(size_t max){
      total = max;
    }

    // --- --- --- Print progress
    void print_progress(std::ostream& out, size_t nbdone){
      if(nbdone>0) {
        const size_t vprev = (nbdone-1)*100/total;
        const size_t vnow = nbdone*100/total;
        const size_t vnow_tenth = vnow/10;
        const size_t vprev_tenth = vprev/10;
        if (vprev<vnow) {
          if (vprev_tenth<vnow_tenth) { out << vnow_tenth*10 << "% "; } else { out << "."; }
          std::flush(out);
        }
      }
    }
  };

}