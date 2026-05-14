#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include "core/Utils.h"
#include <string>
#include <vector>

class LBPFaceRecognitionModule
{
public:
    LBPFaceRecognitionModule();
    ~LBPFaceRecognitionModule();
    void renderUI();

private:
    bool tryLoadCascade(const std::string& path);
    void loadImage(int slot, const std::string& path);  // slot 0=ref, 1=query
    void reprocess();
    void processSlot(int slot);     // detect face + compute LBP + build visuals
    void compareHistograms();
    void exportSlot(int slot);
    void reset();

    // Circular LBP with bilinear interpolation (radius R, P neighbors)
    static cv::Mat computeLBP(const cv::Mat& gray, int radius, int neighbors);

    // Grid-based LBP histogram (LBPH) — returns normalized feature vector
    static std::vector<float> computeLBPHistogram(const cv::Mat& lbp,
                                                   int gridN, int numBins);

    // Chi-square distance between two equal-length histograms
    static float chiSquare(const std::vector<float>& h1,
                           const std::vector<float>& h2);

    // Face detection
    cv::CascadeClassifier m_cascade;
    bool        m_cascadeLoaded = false;
    std::string m_cascadePath;

    // Per-slot state: 0 = reference, 1 = comparison
    cv::Mat     m_rawImg[2];
    cv::Mat     m_faceROI[2];          // 128x128 grayscale face crop
    cv::Rect    m_faceRect[2];         // detected face rect in original coords
    bool        m_hasImage[2] = {false, false};
    bool        m_hasFace[2]  = {false, false};
    std::string m_imgPath[2];
    std::vector<float> m_hist[2];      // LBPH feature vector

    // GPU textures
    Texture m_texRef;        // reference  — annotated original
    Texture m_texQuery;      // comparison — annotated original
    Texture m_texLBPRef;     // reference  — LBP texture with grid
    Texture m_texLBPQuery;   // comparison — LBP texture with grid

    // Cached display images for export
    cv::Mat m_displayRef;
    cv::Mat m_displayQuery;
    cv::Mat m_lbpDisplayRef;
    cv::Mat m_lbpDisplayQuery;

    // LBP parameters
    int m_gridN     = 8;   // NxN spatial grid for histogram
    int m_radius    = 1;   // sampling radius in pixels
    int m_neighbors = 8;   // sampling points on the circle (4 or 8)

    // Haar detection parameters
    float m_detectionScale = 1.1f;   // scaleFactor for detectMultiScale
    int   m_minNeighbors   = 3;      // minNeighbors for detectMultiScale

    // Comparison result
    float m_chiDist     = 0.0f;    // raw chi-square distance
    float m_chiDistNorm = 0.0f;    // normalized (per cell average)
    float m_threshold   = 2.0f;    // "same person" decision boundary
    bool  m_hasComparison = false;

    // Timing and state
    bool  m_needsProcess  = false;
    float m_processTimeMs = 0.0f;

    // Export feedback
    std::string m_exportMsg;
    double      m_exportMsgTime = 0.0;
};
