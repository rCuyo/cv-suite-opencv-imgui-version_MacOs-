#include "HistogramUtils.h"
#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────

std::array<float, 256> HistogramUtils::compute(const cv::Mat& img)
{
    std::array<float, 256> result{};
    if (img.empty()) return result;

    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    const int    histSize   = 256;
    float        range[]    = { 0.0f, 256.0f };
    const float* ranges[]   = { range };
    const int    channels[] = { 0 };

    cv::Mat hist;
    cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, &histSize, ranges);

    // NORM_MINMAX maps the tallest bin to 1.0 and the shortest to 0.0,
    // ensuring binary-image histograms (two extreme peaks) render clearly.
    cv::normalize(hist, hist, 0.0, 1.0, cv::NORM_MINMAX);

    for (int i = 0; i < 256; ++i)
        result[i] = hist.at<float>(i);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────

HistogramUtils::ImageStats HistogramUtils::computeStats(const cv::Mat& img)
{
    ImageStats s;
    if (img.empty()) return s;

    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    s.mean   = static_cast<float>(mean[0]);
    s.stddev = static_cast<float>(stddev[0]);

    const int total = gray.rows * gray.cols;
    if (total > 0) {
        int white = cv::countNonZero(gray == 255);
        int black = total - cv::countNonZero(gray > 0);
        s.whitePct = 100.0f * static_cast<float>(white) / static_cast<float>(total);
        s.blackPct = 100.0f * static_cast<float>(black) / static_cast<float>(total);
    }

    return s;
}
