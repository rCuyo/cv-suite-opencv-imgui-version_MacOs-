#include "HistogramRenderer.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cstdio>

namespace HistogramRenderer
{

cv::Mat render(const std::array<float, 256>& hist,
               const std::string& title,
               const Options& opts)
{
    const int W = opts.width;
    const int H = opts.height;

    // Reserve bottom margin for axis labels
    const int marginBottom = title.empty() ? 10 : 26;
    const int plotH        = H - marginBottom - 6;  // usable bar area height

    cv::Mat canvas(H, W, CV_8UC3,
                   cv::Scalar(opts.bgB, opts.bgG, opts.bgR));

    // Draw a subtle horizontal baseline
    cv::line(canvas,
             cv::Point(0, H - marginBottom),
             cv::Point(W, H - marginBottom),
             cv::Scalar(80, 80, 80), 1);

    // Draw 256 bars (each bar may span multiple pixels horizontally)
    const float barW = static_cast<float>(W) / 256.0f;
    const cv::Scalar barColour(opts.barB, opts.barG, opts.barR);

    for (int i = 0; i < 256; ++i) {
        float v    = std::max(0.0f, std::min(1.0f, hist[i]));
        int   barH = static_cast<int>(v * plotH);
        if (barH == 0) continue;

        int x0 = static_cast<int>(i * barW);
        int x1 = static_cast<int>((i + 1) * barW);
        if (x1 <= x0) x1 = x0 + 1;
        if (x1 > W)   x1 = W;

        int y0 = H - marginBottom - barH;
        int y1 = H - marginBottom;

        cv::rectangle(canvas,
                      cv::Point(x0, y0),
                      cv::Point(x1 - 1, y1 - 1),
                      barColour, cv::FILLED);
    }

    // Draw axis tick labels: 0, 64, 128, 192, 255
    const cv::Scalar tickColour(140, 140, 140);
    const int ticks[] = { 0, 64, 128, 192, 255 };
    for (int t : ticks) {
        int x = static_cast<int>(t * barW);
        cv::line(canvas,
                 cv::Point(x, H - marginBottom),
                 cv::Point(x, H - marginBottom + 3),
                 tickColour, 1);

        char buf[8];
        std::snprintf(buf, sizeof(buf), "%d", t);
        cv::putText(canvas, buf,
                    cv::Point(x - 6, H - marginBottom + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.28,
                    tickColour, 1, cv::LINE_AA);
    }

    // Draw optional title centred at the bottom
    if (!title.empty()) {
        int baseline = 0;
        cv::Size ts  = cv::getTextSize(title,
                                       cv::FONT_HERSHEY_SIMPLEX, 0.38, 1, &baseline);
        int tx = (W - ts.width) / 2;
        cv::putText(canvas, title,
                    cv::Point(tx, H - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }

    return canvas;
}

bool renderAndSave(const std::array<float, 256>& hist,
                   const std::string& outputPath,
                   const std::string& title,
                   const Options& opts)
{
    cv::Mat img = render(hist, title, opts);
    bool ok     = cv::imwrite(outputPath, img);
    if (!ok)
        std::fprintf(stderr, "[HistogramRenderer] No se pudo escribir: %s\n",
                     outputPath.c_str());
    return ok;
}

} // namespace HistogramRenderer
