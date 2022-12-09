#pragma once

#include <tempo/dataset/dts.hpp>

#include <iostream>
#include <optional>
#include <string>

namespace tempo::univariate::writer {

  inline std::optional<std::string> write(
      tempo::DTS const& split,
      std::string const& problem_name,
      std::ostream& out){

    if(split.size() == 0){ return {"Can't write empty split"}; }

    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);

    out << "# File produced by the tempo TS writer" << std::endl;
    out << "# " << std::ctime(&now_time) << std::endl;

    out << "@problemname " << problem_name << std::endl;
    out << "@timestamps false" << std::endl;
    out << "@univariate true" << std::endl;
    out << "@targetlabel false" << std::endl;

    // Check for missing values
    bool has_missing = false;
    for(size_t i=0; i<split.size(); ++i){
      if(split[i].missing()){
        has_missing = true;
        break;
      }
    }

    out << "@missing " << ( has_missing?"true":"false" ) << std::endl;

    // Check for equal length
    const size_t length0 = split[0].length();
    bool equal_length = true;
    for(size_t i=1; i<split.size(); ++i){
      if(split[i].length() != length0){
        equal_length = false;
        break;
      }
    }

    out << "@equallength " << ( equal_length?"true":"false" ) << std::endl;
    if(equal_length){
      out << "@serieslength " << length0 << std::endl;
    }


    // Dataset labels
    const auto& vec_labels = split.header().label_encoder().index_to_label();
    if(vec_labels.empty()){
      out << "@classlabel false" << std::endl;
    } else {
      out << "@classlabel true";
      for(const auto& l:vec_labels){ out << " " << l; }
      out << std::endl;
    }

    // Data section
    out << "@data" << std::endl;
    out.precision(17);

    for(size_t i=0; i<split.size(); ++i){
      const auto& ts = split[i];
      // Univariate data
      out << ts[0];
      for(size_t j=1; j<ts.length(); ++j){ out << "," << ts[j]; }
      // Label if we have one
      if(ts.label()){ out << ":" << ts.label().value(); }
      out << std::endl;
    }



    return {};

  }

}