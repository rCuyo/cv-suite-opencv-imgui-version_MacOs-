#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include "core/Utils.h"
#include <atomic>
#include <string>
#include <thread>
#include <vector>

enum class HOGInputMode { Image, Video };

class HOGPedestrianModule
{
public:
    HOGPedestrianModule();
    ~HOGPedestrianModule();
    void renderUI();

private:
    // ── Mode ──────────────────────────────────────────────────────────────────
    HOGInputMode m_mode = HOGInputMode::Image;

    void renderImageMode();
    void renderVideoMode();
    void renderParameters();   // shared by both modes

    // ── Shared HOG parameters ─────────────────────────────────────────────────
    double m_hitThreshold   = 0.0;
    int    m_winStrideX     = 8;
    int    m_winStrideY     = 8;
    double m_scale          = 1.05;
    int    m_groupThreshold = 2;

    // ── HOG descriptor (image mode — UI thread only) ──────────────────────────
    cv::HOGDescriptor m_hog;

    // ── IMAGE MODE ────────────────────────────────────────────────────────────
    void loadImage(const std::string& path);
    void detectOnImage();
    void drawImageRow();

    cv::Mat     m_imageBGR;
    cv::Mat     m_imageProcessed;
    Texture     m_texImageOrig;
    Texture     m_texImageResult;
    std::string m_imagePath;
    int         m_imageDetCount    = 0;
    bool        m_imageNeedsDetect = false;
    float       m_imageDetectMs    = 0.0f;
    std::string m_imageExportMsg;
    double      m_imageExportTime  = 0.0;

    // ── VIDEO MODE ────────────────────────────────────────────────────────────
    void loadVideoMeta(const std::string& path);
    void startProcessing();
    void cancelProcessing();
    void workerProcess();   // runs on m_workerThread
    void pollWorker();      // called each render frame to check completion

    std::string m_videoInputPath;
    std::string m_videoOutputDir;   // "" → use exports/
    bool        m_hasVideoInput = false;

    // Worker thread + atomic communication
    std::thread       m_workerThread;
    std::atomic<bool> m_isProcessing{false};
    std::atomic<bool> m_cancelRequested{false};
    std::atomic<int>  m_processedFrames{0};
    std::atomic<int>  m_detectedTotal{0};

    // Video metadata (populated before thread starts — read-only after)
    int       m_vidTotalFrames   = 0;
    int       m_vidWidth         = 0;
    int       m_vidHeight        = 0;
    double    m_vidFps           = 0.0;
    double    m_vidDurationSec   = 0.0;
    uintmax_t m_vidFileSizeBytes = 0;

    // Worker result (written by worker, read by UI after join)
    bool        m_workerSuccess    = false;
    std::string m_workerOutputPath;
    std::string m_workerError;
    float       m_workerElapsedSec = 0.0f;

    // UI-only state
    bool   m_showResult   = false;
    double m_startTimeSec = 0.0;
};
