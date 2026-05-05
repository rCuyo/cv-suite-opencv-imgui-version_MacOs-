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
    EdgeDetectionModule();
    ~EdgeDetectionModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void reprocess();
    void processOCR(const cv::Mat& gray);
    void processSegmentation(const cv::Mat& gray);
    void drawImageRow();

    EdgeMode m_mode        = EdgeMode::OCRDocument;
    bool     m_needsProcess = false;

    cv::Mat m_original;
    cv::Mat m_gray;
    cv::Mat m_edges;        // Canny output (single channel)
    cv::Mat m_contourViz;   // Contours drawn on colour copy (segmentation mode)

    Texture m_texOrig;
    Texture m_texEdges;
    Texture m_texContours; // Only valid in Segmentation mode

    // ── Canny parameters ──────────────────────────────────────────────────────
    int m_gaussKernel  = 5;  // Kernel size for Gaussian pre-blur (odd, >= 3)
    int m_cannyLow     = 50; // Hysteresis lower threshold
    int m_cannyHigh    = 150;// Hysteresis upper threshold
    int m_apertureSize = 3;  // Sobel aperture (3, 5, or 7)

    // ── Segmentation parameters ───────────────────────────────────────────────
    int m_minContourArea = 200; // Ignore contours smaller than this (px²)

    char m_pathBuf[512];
};
