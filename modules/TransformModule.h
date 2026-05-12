#pragma once
#include <opencv2/core.hpp>
#include "core/Utils.h"
#include "core/HistogramUtils.h"
#include <array>
#include <string>

enum class TransformMode {
    Translation,  // Shift image along X/Y axes via warpAffine
    Rotation,     // Rotate image around its center via getRotationMatrix2D
    Scaling       // Resize image with independent X/Y factors via cv::resize
};

class TransformModule
{
public:
    TransformModule()  = default;
    ~TransformModule();

    void renderUI();
    void loadAndProcess(const std::string& path);

private:
    void reprocess();
    void applyTranslation();
    void applyRotation();
    void applyScaling();
    void drawImageRow();
    void drawHistograms();

    // ── State ─────────────────────────────────────────────────────────────────
    TransformMode m_mode         = TransformMode::Translation;
    bool          m_needsProcess = false;
    std::string   m_loadedPath;

    cv::Mat m_original;  // source image, never modified
    cv::Mat m_result;    // warpAffine output

    Texture m_texOrig;
    Texture m_texResult;

    std::array<float, 256>     m_histOrig{};
    std::array<float, 256>     m_histResult{};
    HistogramUtils::ImageStats m_statsOrig{};
    HistogramUtils::ImageStats m_statsResult{};

    // ── Translation ───────────────────────────────────────────────────────────
    float m_translationX    = 0.0f;
    float m_translationY    = 0.0f;
    bool  m_isDragging      = false;
    float m_dragStartMouseX = 0.0f;
    float m_dragStartMouseY = 0.0f;
    float m_dragStartTX     = 0.0f;
    float m_dragStartTY     = 0.0f;

    // ── Rotation ──────────────────────────────────────────────────────────────
    float m_angle = 0.0f;

    // ── Scaling ───────────────────────────────────────────────────────────────
    float m_scaleX        = 1.0f;
    float m_scaleY        = 1.0f;
    bool  m_uniformScale  = true;   // single slider forces X == Y
    bool  m_lockAspect    = true;   // independent sliders keep h/w ratio

    // ── Export feedback ───────────────────────────────────────────────────────
    std::string m_exportMsg;
    double      m_exportMsgTime = 0.0;
};
