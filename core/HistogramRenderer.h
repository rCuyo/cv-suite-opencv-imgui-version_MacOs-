#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <string>

// Renders a 256-bin normalized histogram as a cv::Mat image using OpenCV
// drawing primitives (cv::line, cv::rectangle).  The result can be saved
// directly with cv::imwrite — no screenshots involved.
namespace HistogramRenderer
{
    struct Options {
        int   width      = 512;
        int   height     = 200;
        // Bar colour (BGR)
        uchar barB       = 180;
        uchar barG       = 180;
        uchar barR       = 255;
        // Background colour (BGR)
        uchar bgB        = 30;
        uchar bgG        = 30;
        uchar bgR        = 30;
    };

    // Render a normalized histogram (values in [0,1]) to a BGR image.
    cv::Mat render(const std::array<float, 256>& hist,
                   const std::string& title = "",
                   const Options& opts = {});

    // Convenience: render and save immediately.
    // Returns true on success.
    bool renderAndSave(const std::array<float, 256>& hist,
                       const std::string& outputPath,
                       const std::string& title = "",
                       const Options& opts = {});
}
