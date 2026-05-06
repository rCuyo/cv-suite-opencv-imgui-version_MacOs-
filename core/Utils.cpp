// OpenGL 3.3 — plataforma cruzada (glad en Windows, gl3.h en macOS)
#include "gl.h"

#include "Utils.h"
#include <opencv2/imgproc.hpp>

// ─── Conversion: cv::Mat → OpenGL texture ────────────────────────────────────
//
// Key steps:
//   1. Normalise channel order: OpenCV stores BGR; OpenGL expects RGB.
//   2. Ensure contiguous rows (necessary for glTexImage2D).
//   3. Generate a GL_TEXTURE_2D object and upload pixel data.
//
// The result is suitable for ImGui::Image():
//   ImGui::Image((ImTextureID)(uintptr_t)tex.id, ImVec2(tex.width, tex.height));

Texture Utils::matToTexture(const cv::Mat& mat)
{
    if (mat.empty()) return {};

    // ── Convert to RGB ────────────────────────────────────────────────────────
    cv::Mat rgb;
    switch (mat.channels())
    {
    case 1: cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB); break;
    case 3: cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);  break;
    case 4: cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB); break;
    default: mat.copyTo(rgb);
    }

    // Ensure the data is stored in a contiguous block (no row padding)
    if (!rgb.isContinuous())
        rgb = rgb.clone();

    // ── Upload to GPU ─────────────────────────────────────────────────────────
    Texture tex;
    tex.width  = rgb.cols;
    tex.height = rgb.rows;

    glGenTextures(1, &tex.id);
    glBindTexture(GL_TEXTURE_2D, tex.id);

    // Bilinear filtering; clamp to edge so border pixels are not repeated
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // RGB rows may not be 4-byte aligned
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8,
                 rgb.cols, rgb.rows, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, rgb.data);

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

void Utils::updateTexture(Texture& tex, const cv::Mat& mat)
{
    deleteTexture(tex);
    tex = matToTexture(mat);
}

void Utils::deleteTexture(Texture& tex)
{
    if (tex.id != 0) {
        glDeleteTextures(1, &tex.id);
        tex = {};
    }
}
