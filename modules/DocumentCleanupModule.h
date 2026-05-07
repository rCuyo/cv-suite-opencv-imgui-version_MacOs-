#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include "core/HistogramUtils.h"
#include <array>
#include <string>
#include <vector>

class DocumentCleanupModule
{
public:
    DocumentCleanupModule()  = default;
    ~DocumentCleanupModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void    reprocess();
    bool    detectPageAndWarp(const cv::Mat& bgr, cv::Mat& out);
    cv::Mat applyCleanup(const cv::Mat& gray);
    void    drawImageRow();
    void    drawHistograms();

    // ── State ─────────────────────────────────────────────────────────────────
    bool        m_needsProcess = false;
    bool        m_warpApplied  = false;
    std::string m_loadedPath;

    cv::Mat m_rawBGR;   // image as loaded from disk — never modified
    cv::Mat m_original; // displayed "Original" (may be perspective-corrected)
    cv::Mat m_result;   // pipeline output

    Texture m_texOrig;
    Texture m_texResult;

    std::array<float, 256>      m_histOrig{};
    std::array<float, 256>      m_histResult{};
    HistogramUtils::ImageStats  m_statsOrig{};
    HistogramUtils::ImageStats  m_statsResult{};

    // ── Processing parameters ────────────────────────────────────────────────
    int   m_blurKernel      = 3;
    float m_claheClip       = 2.0f;
    int   m_adaptBlockSize  = 21;
    int   m_adaptC          = 10;
    float m_sharpenStrength = 0.5f;

    // ── Toggles ──────────────────────────────────────────────────────────────
    bool m_perspCorrection = true;
    bool m_shadowReduction = true;
    bool m_sharpenEnabled  = true;
    bool m_binaryOut       = true;
    bool m_morphCleanup    = false;
};
