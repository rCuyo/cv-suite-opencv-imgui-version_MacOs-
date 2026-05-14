#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "core/Utils.h"
#include <string>
#include <vector>

class ORBRecognitionModule
{
public:
    ORBRecognitionModule()  = default;
    ~ORBRecognitionModule();
    void renderUI();

private:
    void loadReference(const std::string& path);
    void loadQuery(const std::string& path);
    void reprocess();
    void detectFeatures();
    void matchFeatures();
    void buildVisuals();
    void exportImage(const cv::Mat& img, const std::string& prefix);
    void reset();

    // Source images — never modified after load
    cv::Mat m_rawRef;
    cv::Mat m_rawQuery;

    // ORB output
    std::vector<cv::KeyPoint> m_kpRef;
    std::vector<cv::KeyPoint> m_kpQuery;
    cv::Mat                   m_descRef;
    cv::Mat                   m_descQuery;
    std::vector<cv::DMatch>   m_goodMatches;

    // Display images (scaled BGR, drawn with keypoints/matches)
    cv::Mat m_imgRefDisplay;
    cv::Mat m_imgQueryDisplay;
    cv::Mat m_imgMatches;

    // GPU textures
    Texture m_texRef;
    Texture m_texQuery;
    Texture m_texMatches;

    // ORB parameters
    int   m_nFeatures     = 500;
    float m_scaleFactor   = 1.2f;
    int   m_nLevels       = 8;
    int   m_matchThreshold = 60;  // max Hamming distance (0-256)

    // Paths (for UI display)
    std::string m_refPath;
    std::string m_queryPath;

    // State flags
    bool m_hasRef       = false;
    bool m_hasQuery     = false;
    bool m_hasFeatures  = false;
    bool m_hasMatches   = false;
    bool m_needsProcess = false;

    // Statistics
    int   m_kpRefCount   = 0;
    int   m_kpQueryCount = 0;
    int   m_matchCount   = 0;
    float m_detectTimeMs = 0.0f;
    float m_matchTimeMs  = 0.0f;

    // Export feedback (timed message pattern)
    std::string m_exportMsg;
    double      m_exportMsgTime = 0.0;
};
