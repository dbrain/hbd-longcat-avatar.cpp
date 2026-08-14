#pragma once

#include <filesystem>

namespace visp {
using path = std::filesystem::path;

path current_library_path();

} // namespace visp
