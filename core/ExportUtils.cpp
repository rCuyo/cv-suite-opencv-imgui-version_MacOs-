#include "ExportUtils.h"
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace ExportUtils
{

std::string nextExportPath(const std::string& dir,
                           const std::string& prefix,
                           const std::string& ext)
{
    fs::path base = dir.empty() ? fs::current_path() : fs::path(dir);

    // Create output directory if it doesn't exist yet
    if (!dir.empty())
        fs::create_directories(base);

    for (int n = 1; n < 10000; ++n) {
        fs::path candidate = base / (prefix + "_" + std::to_string(n) + "." + ext);
        if (!fs::exists(candidate))
            return candidate.string();
    }

    // Fallback — should never be reached in practice
    return (base / (prefix + "_overflow." + ext)).string();
}

} // namespace ExportUtils
