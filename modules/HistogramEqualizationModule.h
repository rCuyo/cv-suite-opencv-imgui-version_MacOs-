#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include "core/HistogramUtils.h"
#include <array>
#include <string>

class HistogramEqualizationModule
{
public:
    HistogramEqualizationModule()  = default;
    ~HistogramEqualizationModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void reprocess();
    void drawImageRow();
    void drawHistograms();

    // ── State ─────────────────────────────────────────────────────────────────
    bool        m_needsProcess = false;
    std::string m_loadedPath;

    cv::Mat m_original;  // BGR as loaded — never modified
    cv::Mat m_gray;      // grayscale of original
    cv::Mat m_equalized; // cv::equalizeHist output

    Texture m_texGray;
    Texture m_texEqualized;

    std::array<float, 256>     m_histOrig{};
    std::array<float, 256>     m_histEqualized{};
    HistogramUtils::ImageStats m_statsOrig{};
    HistogramUtils::ImageStats m_statsEqualized{};

    // ── Export feedback ───────────────────────────────────────────────────────
    std::string m_exportMsg;
    double      m_exportMsgTime = 0.0;
};
