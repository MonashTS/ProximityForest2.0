#pragma once

#include <optional>
#include <string>

/// Exit with a code and a message
[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {});
