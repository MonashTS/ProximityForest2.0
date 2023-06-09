#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/utils/label_encoder.hpp>
#include <tempo/reader/reader_result.hpp>

#include <istream>
#include <set>


namespace tempo::reader::univariate {

  enum CSVLabel { NONE, FIRST, LAST };

  struct CSVReaderParam {
    CSVLabel label_position;
    bool has_header;
    char field_sep;
    std::set<char> comment_skip;

    explicit CSVReaderParam(CSVLabel lp, bool has_header = false, char fsep = ',', std::set<char> cskip = {'#'}) :
      label_position(lp), has_header(has_header), field_sep(fsep), comment_skip(std::move(cskip)) {}
  };

  template<std::floating_point F>
  Result<F> read_csv(std::istream& input, LabelEncoder const& other, CSVReaderParam const& params);

} // End of namespace tempo::reader

