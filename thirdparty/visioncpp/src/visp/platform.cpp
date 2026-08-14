#include "visp/platform.h"

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace visp {
using path = std::filesystem::path;

path current_library_path() {

#ifdef _WIN32
    HMODULE module = nullptr;
    DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (GetModuleHandleExW(flags, (LPCWSTR)&current_library_path, &module)) {
        wchar_t buffer[MAX_PATH];
        if (GetModuleFileNameW(module, buffer, MAX_PATH) > 0) {
            return std::filesystem::path(buffer);
        }
    }
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&current_library_path), &info)) {
        return std::filesystem::path(info.dli_fname);
    }
#endif
    return std::filesystem::path();
}

} // namespace visp
