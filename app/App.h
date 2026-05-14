#pragma once
#include <memory>
#include <string>

// Forward-declare GLFW handle to avoid pulling OpenGL headers into every TU
struct GLFWwindow;

class ThresholdModule;
class EdgeDetectionModule;
class DocumentCleanupModule;
class TransformModule;
class HistogramEqualizationModule;
class NoiseReductionModule;
class MorphologyModule;
class HOGPedestrianModule;
class SIFTRecognitionModule;
class ORBRecognitionModule;
class LBPFaceRecognitionModule;

enum class ActiveModule { None, Threshold, EdgeDetection, DocumentCleanup, Transform, HistogramEq, NoiseReduction, Morphology, HOGPedestrian, SIFTRecognition, ORBRecognition, LBPFaceRecognition };

class App
{
public:
    App(const std::string& title, int width, int height);
    ~App();

    bool init();
    void run();

private:
    void renderUI();
    void renderSidebar();
    void renderContent();
    void shutdown();

    std::string   m_title;
    int           m_width;
    int           m_height;
    GLFWwindow*   m_window       = nullptr;
    ActiveModule  m_activeModule = ActiveModule::None;

    std::unique_ptr<ThresholdModule>       m_thresholdModule;
    std::unique_ptr<EdgeDetectionModule>   m_edgeModule;
    std::unique_ptr<DocumentCleanupModule> m_docCleanupModule;
    std::unique_ptr<TransformModule>             m_transformModule;
    std::unique_ptr<HistogramEqualizationModule> m_histEqModule;
    std::unique_ptr<NoiseReductionModule>        m_noiseModule;
    std::unique_ptr<MorphologyModule>            m_morphologyModule;
    std::unique_ptr<HOGPedestrianModule>         m_hogModule;
    std::unique_ptr<SIFTRecognitionModule>       m_siftModule;
    std::unique_ptr<ORBRecognitionModule>        m_orbModule;
    std::unique_ptr<LBPFaceRecognitionModule>    m_lbpModule;
};
