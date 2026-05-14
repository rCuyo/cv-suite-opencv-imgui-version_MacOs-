#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "core/Utils.h"
#include <string>
#include <vector>

class SIFTRecognitionModule
{
public:
    SIFTRecognitionModule()  = default;
    ~SIFTRecognitionModule();
    void renderUI();

private:
    void loadReference(const std::string& path);
    void loadQuery(const std::string& path);
    void reprocess();       // detect + match + build visuals
    void detectFeatures();  // SIFT on both images
    void matchFeatures();   // BFMatcher + Lowe ratio test
    void buildVisuals();    // drawKeypoints + drawMatches → GPU textures
    void exportImage(const cv::Mat& img, const std::string& prefix);
    void reset();

    // Source images — never modified after load
    cv::Mat m_rawRef;
    cv::Mat m_rawQuery;

    // SIFT output
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

    // SIFT parameters
    int   m_nFeatures         = 500;
    float m_contrastThreshold = 0.04f;
    float m_edgeThreshold     = 10.0f;

    // Lowe ratio test threshold
    float m_ratioThresh = 0.75f;

    // Paths (for display)
    std::string m_refPath;
    std::string m_queryPath;

    // State flags
    bool m_hasRef       = false;
    bool m_hasQuery     = false;
    bool m_hasFeatures  = false;  // both images have descriptors
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
