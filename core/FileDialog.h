#pragma once
#include <string>

namespace FileDialog
{
    // Show an "Open" file picker filtered to common image formats.
    // Returns the selected path, or empty string if cancelled.
    std::string openImage(const std::string& defaultPath = "");

    // Show an "Open" file picker filtered to common video formats (mp4, avi, mov, mkv).
    // Returns the selected path, or empty string if cancelled.
    std::string openVideo(const std::string& defaultPath = "");

    // Show a folder picker dialog.
    // Returns the selected folder path, or empty string if cancelled.
    std::string openFolder(const std::string& defaultPath = "");

    // Show a "Save" file picker for PNG or JPG output.
    // Returns the destination path, or empty string if cancelled.
    std::string saveImage(const std::string& defaultPath = "result.png");
}
