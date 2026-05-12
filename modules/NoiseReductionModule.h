#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include "core/HistogramUtils.h"
#include <array>
#include <string>

enum class NoiseFilterMode {
    Gaussian,   // cv::GaussianBlur  — linear convolution, general smoothing
    Median,     // cv::medianBlur    — non-linear, robust to salt-and-pepper
    Bilateral   // cv::bilateralFilter — edge-preserving, non-linear
};

class NoiseReductionModule
{
public:
    NoiseReductionModule()  = default;
    ~NoiseReductionModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void reprocess();
    void applyGaussian();
    void applyMedian();
    void applyBilateral();
    void drawImageRow();
    void drawHistograms();

    // ── State ─────────────────────────────────────────────────────────────────
    NoiseFilterMode m_mode          = NoiseFilterMode::Gaussian;
    bool            m_needsProcess  = false;
    float           m_lastProcessMs = 0.0f;
    std::string     m_loadedPath;

    cv::Mat m_original;
    cv::Mat m_result;

    Texture m_texOrig;
    Texture m_texResult;

    std::array<float, 256>     m_histOrig{};
    std::array<float, 256>     m_histResult{};
    HistogramUtils::ImageStats m_statsOrig{};
    HistogramUtils::ImageStats m_statsResult{};

    // ── Gaussian parameters ───────────────────────────────────────────────────
    int   m_gaussKernel = 5;    // must be odd ≥ 1
    float m_gaussSigma  = 0.0f; // 0 = auto-computed from kernel size

    // ── Median parameters ─────────────────────────────────────────────────────
    int m_medianKernel = 5;     // must be odd ≥ 1

    // ── Bilateral parameters ──────────────────────────────────────────────────
    int   m_bilateralD          =  9;
    float m_bilateralSigmaColor = 75.0f;
    float m_bilateralSigmaSpace = 75.0f;

    // ── Export feedback ───────────────────────────────────────────────────────
    std::string m_exportMsg;
    double      m_exportMsgTime = 0.0;
};
