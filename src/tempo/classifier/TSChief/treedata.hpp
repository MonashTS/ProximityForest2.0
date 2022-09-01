#pragma once

#include <any>
#include <memory>
#include <utility>
#include <vector>
#include "tempo/classifier/utils.hpp"

namespace tempo::classifier::TSChief {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Data management

  struct TreeData;

  template<typename Data>
  struct i_GetData {
    virtual ~ i_GetData() = default;
    virtual Data const& at(TreeData const& td) = 0;
  };

  struct TreeData {
    // --- --- --- Types
    template<typename Data>
    struct GetData : i_GetData<Data> {
      size_t index;
      explicit GetData(size_t idx) : index(idx) {}
      Data const& at(TreeData const& td) override { return *std::static_pointer_cast<Data>(td.data[index]).get(); }
    };

    // --- --- --- Fields
    std::vector<std::shared_ptr<void>> data{};

    // --- --- --- Constructor/Destructor

    // --- --- --- Methods
    template<typename Data>
    std::shared_ptr<i_GetData<Data>> register_data(std::shared_ptr<void>&& sptr) {
      size_t idx = data.size();
      data.push_back(std::move(sptr));
      return std::make_shared<GetData<Data>>(idx);
    }
  };

} // End of tempo::classifier::TSChief