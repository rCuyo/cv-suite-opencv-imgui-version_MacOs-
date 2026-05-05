#pragma once
#include <opencv2/core.hpp>

// Use unsigned int so this header stays free of OpenGL headers.
// GLuint is always uint32_t on macOS / desktop GL.
struct Texture
{
    unsigned int id     = 0;
    int          width  = 0;
    int          height = 0;

    bool valid() const { return id != 0 && width > 0 && height > 0; }
};

namespace Utils
{
    // Upload a cv::Mat (any channel count) to the GPU and return a Texture handle.
    // Caller is responsible for calling deleteTexture() when done.
    Texture matToTexture(const cv::Mat& mat);

    // Replace the GPU image with new pixel data (deletes old, creates new).
    void updateTexture(Texture& tex, const cv::Mat& mat);

    // Free the GPU texture.
    void deleteTexture(Texture& tex);
}
