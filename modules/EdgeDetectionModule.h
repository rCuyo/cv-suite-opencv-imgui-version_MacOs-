#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include <string>

enum class EdgeMode {
    OCRDocument,   // Canny on a document image; compare with thresholding
    Segmentation   // Extract and draw object contours
};

class EdgeDetectionModule
{
public:
    EdgeDetectionModule()  = default;
    ~EdgeDetectionModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void reprocess();
    void processOCR(const cv::Mat& gray);
    void processSegmentation(const cv::Mat& gray);
    void drawImageRow();

    EdgeMode    m_mode         = EdgeMode::OCRDocument;
    bool        m_needsProcess = false;
    std::string m_loadedPath;

    cv::Mat m_original;
    cv::Mat m_gray;
    cv::Mat m_edges;
    cv::Mat m_contourViz;

    Texture m_texOrig;
    Texture m_texEdges;
    Texture m_texContours;

    // ── Canny parameters ──────────────────────────────────────────────────────
    int m_gaussKernel  = 5;
    int m_cannyLow     = 50;
    int m_cannyHigh    = 150;
    int m_apertureSize = 3;

    // ── Segmentation parameters ───────────────────────────────────────────────
    int m_minContourArea = 200;
};
