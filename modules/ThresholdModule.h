#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include <string>

enum class ThresholdMode {
    OCR,                 // Binarise documents for text recognition
    MedicalSegmentation, // Segment tissue regions in grayscale scans
    IndustrialInspection // Detect surface defects / missing parts
};

class ThresholdModule
{
public:
    ThresholdModule();
    ~ThresholdModule();

    // Called by App every frame the module is active
    void renderUI();

    // Load an image from the filesystem and process it
    void loadAndProcess(const std::string& path);

private:
    // ── Processing ────────────────────────────────────────────────────────────
    void reprocess();
    void processOCR(const cv::Mat& gray);
    void processMedical(const cv::Mat& gray);
    void processIndustrial(const cv::Mat& gray);

    // ── Helpers ───────────────────────────────────────────────────────────────
    void uploadTextures();
    void drawImageRow();    // Render original + result(s) side by side

    // ── State ─────────────────────────────────────────────────────────────────
    ThresholdMode m_mode = ThresholdMode::OCR;
    bool m_needsProcess  = false;

    cv::Mat m_original;   // Color original stored for re-display
    cv::Mat m_gray;       // Grayscale copy (source for all processing)
    cv::Mat m_resultA;    // Primary result  (Otsu / segmented / defects)
    cv::Mat m_resultB;    // Secondary result (Adaptive; unused in other modes)

    Texture m_texOrig;
    Texture m_texA;
    Texture m_texB;       // Only valid in OCR mode

    std::string m_labelA;
    std::string m_labelB;

    // ── OCR parameters ────────────────────────────────────────────────────────
    int   m_adaptiveBlockSize = 11; // Must be odd, >= 3
    int   m_adaptiveC         = 2;  // Constant subtracted from weighted mean

    // ── Medical parameters ────────────────────────────────────────────────────
    int m_medLow  = 50;
    int m_medHigh = 180;

    // ── Industrial parameters ─────────────────────────────────────────────────
    int m_defectThresh  = 40; // Pixels darker than this are flagged as defects
    int m_gaussBlurSize = 5;  // Blur before thresholding to remove sensor noise

    // ImGui input buffer for file path
    char m_pathBuf[512];
};
