#include "FileDialog.h"
#include "tinyfiledialogs.h"

std::string FileDialog::openImage(const std::string& defaultPath)
{
    static const char* filters[] = { "*.png", "*.jpg", "*.jpeg", "*.bmp" };
    const char* result = tinyfd_openFileDialog(
        "Open Image",
        defaultPath.empty() ? "" : defaultPath.c_str(),
        4, filters,
        "Image files (png, jpg, jpeg, bmp)",
        0);
    return result ? result : "";
}

std::string FileDialog::saveImage(const std::string& defaultPath)
{
    static const char* filters[] = { "*.png", "*.jpg" };
    const char* result = tinyfd_saveFileDialog(
        "Save Image",
        defaultPath.empty() ? "result.png" : defaultPath.c_str(),
        2, filters,
        "Image files (png, jpg)");
    return result ? result : "";
}
