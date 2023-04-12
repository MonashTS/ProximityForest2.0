#pragma once

#include "pch.h"
#include <tempo/utils/simplecli.hpp>


extern std::string usage;

/// Exit with a code and a message
[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {});