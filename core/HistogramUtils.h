#pragma once
#include <opencv2/core.hpp>
#include <array>

namespace HistogramUtils
{
    // Normalized histogram: 256 bins, values in [0, 1] via cv::NORM_MINMAX.
    // Multi-channel inputs are converted to grayscale before computing.
    std::array<float, 256> compute(const cv::Mat& img);

    struct ImageStats {
        float mean     = 0.0f;  // Mean intensity [0, 255]
        float stddev   = 0.0f;  // Standard deviation
        float whitePct = 0.0f;  // % of pixels at intensity 255
        float blackPct = 0.0f;  // % of pixels at intensity 0
    };

    // Compute per-image statistics (mean, stddev, white/black pixel %).
    // Multi-channel inputs are converted to grayscale before computing.
    ImageStats computeStats(const cv::Mat& img);
}
