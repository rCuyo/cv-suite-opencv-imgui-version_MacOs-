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

std::string FileDialog::openVideo(const std::string& defaultPath)
{
    static const char* filters[] = { "*.mp4", "*.avi", "*.mov", "*.mkv" };
    const char* result = tinyfd_openFileDialog(
        "Abrir Video",
        defaultPath.empty() ? "" : defaultPath.c_str(),
        4, filters,
        "Archivos de video (mp4, avi, mov, mkv)",
        0);
    return result ? result : "";
}

std::string FileDialog::openFolder(const std::string& defaultPath)
{
    const char* result = tinyfd_selectFolderDialog(
        "Elegir carpeta de salida",
        defaultPath.empty() ? "" : defaultPath.c_str());
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
