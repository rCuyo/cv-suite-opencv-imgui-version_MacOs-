#include "ImageLoader.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>

cv::Mat ImageLoader::load(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty())
        std::fprintf(stderr, "[ImageLoader] Could not read: %s\n", path.c_str());
    return img;
}

cv::Mat ImageLoader::loadGray(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        std::fprintf(stderr, "[ImageLoader] Could not read: %s\n", path.c_str());
    return img;
}
