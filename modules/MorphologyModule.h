#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include "core/HistogramUtils.h"
#include <array>
#include <string>

enum class MorphOp {
    Erosion,   // cv::erode()                        — shrinks bright regions
    Dilation,  // cv::dilate()                       — expands bright regions
    Opening,   // cv::morphologyEx MORPH_OPEN        — erosion then dilation
    Closing    // cv::morphologyEx MORPH_CLOSE       — dilation then erosion
};

enum class KernelShape {
    Rectangle, // MORPH_RECT
    Ellipse,   // MORPH_ELLIPSE
    Cross      // MORPH_CROSS
};

class MorphologyModule
{
public:
    MorphologyModule()  = default;
    ~MorphologyModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void reprocess();
    void applyOperation();
    void drawImageRow();
    void drawHistograms();

    // ── State ─────────────────────────────────────────────────────────────────
    MorphOp     m_op            = MorphOp::Erosion;
    KernelShape m_kernelShape   = KernelShape::Rectangle;
    bool        m_needsProcess  = false;
    float       m_lastProcessMs = 0.0f;
    std::string m_loadedPath;

    cv::Mat m_original;
    cv::Mat m_result;

    Texture m_texOrig;
    Texture m_texResult;

    std::array<float, 256>     m_histOrig{};
    std::array<float, 256>     m_histResult{};
    HistogramUtils::ImageStats m_statsOrig{};
    HistogramUtils::ImageStats m_statsResult{};

    // ── Parameters ────────────────────────────────────────────────────────────
    int m_kernelSize = 3;   // odd, 1–21
    int m_iterations = 1;   // 1–10

    // ── Export feedback ───────────────────────────────────────────────────────
    std::string m_exportMsg;
    double      m_exportMsgTime = 0.0;
};
