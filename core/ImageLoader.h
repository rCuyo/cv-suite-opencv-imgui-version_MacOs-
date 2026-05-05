#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace ImageLoader
{
    // Load a BGR image from disk; returns empty Mat on failure.
    cv::Mat load(const std::string& path);

    // Load and immediately convert to grayscale.
    cv::Mat loadGray(const std::string& path);
}
