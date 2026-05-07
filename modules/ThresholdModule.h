#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include "core/HistogramUtils.h"
#include <array>
#include <string>
#include <vector>

enum class ThresholdMode {
    OCR,                 // Binarise documents for text recognition
    MedicalSegmentation, // Segment tissue regions in grayscale scans
    IndustrialInspection // Detect surface defects / missing parts
};

class ThresholdModule
{
public:
    ThresholdModule()  = default;
    ~ThresholdModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void reprocess();
    void processOCR(const cv::Mat& gray);
    void processMedical(const cv::Mat& gray);
    void processIndustrial(const cv::Mat& gray);

    void uploadTextures();
    void drawImageRow();
    void drawHistograms();

    // ── State ─────────────────────────────────────────────────────────────────
    ThresholdMode m_mode        = ThresholdMode::OCR;
    bool          m_needsProcess = false;
    std::string   m_loadedPath;

    cv::Mat m_original;
    cv::Mat m_gray;
    cv::Mat m_resultA;
    cv::Mat m_resultB;

    Texture m_texOrig;
    Texture m_texA;
    Texture m_texB;

    std::string m_labelA;
    std::string m_labelB;

    // ── Histograms + statistics ───────────────────────────────────────────────
    std::array<float, 256>       m_histOrig{};
    std::array<float, 256>       m_histA{};
    std::array<float, 256>       m_histB{};
    HistogramUtils::ImageStats   m_statsOrig{};
    HistogramUtils::ImageStats   m_statsA{};
    HistogramUtils::ImageStats   m_statsB{};

    // ── OCR parameters ────────────────────────────────────────────────────────
    int m_adaptiveBlockSize = 11;
    int m_adaptiveC         = 2;

    // ── Medical parameters ────────────────────────────────────────────────────
    int m_medLow  = 50;
    int m_medHigh = 180;

    // ── Industrial parameters ─────────────────────────────────────────────────
    int m_defectThresh  = 40;
    int m_gaussBlurSize = 5;
};
