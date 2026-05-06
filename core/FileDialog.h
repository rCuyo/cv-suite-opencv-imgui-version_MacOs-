#pragma once
#include <string>

namespace FileDialog
{
    // Show an "Open" file picker filtered to common image formats.
    // Returns the selected path, or empty string if cancelled.
    std::string openImage(const std::string& defaultPath = "");

    // Show a "Save" file picker for PNG or JPG output.
    // Returns the destination path, or empty string if cancelled.
    std::string saveImage(const std::string& defaultPath = "result.png");
}
