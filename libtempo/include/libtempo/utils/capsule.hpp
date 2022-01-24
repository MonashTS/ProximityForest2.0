#pragma once

#include <any>
#include <memory>

namespace libtempo::utils {

  /// Capsule
  using Capsule = std::shared_ptr<std::any>;

  /// Capsule builder helper
  template<typename T, typename... Args>
  [[nodiscard]] inline Capsule make_capsule(Args &&... args) {
    return std::make_shared<std::any>(std::make_any<T>(args...));
  }

  /// Capsule pointer accessor
  template<typename T>
  [[nodiscard]] inline T *get_capsule_ptr(const std::shared_ptr <std::any> &capsule) {
    return std::any_cast<T>(capsule.get());
  }


} // End of namespace libtempo::utils
