#pragma once
#include <memory>
#include <string>

// Forward-declare GLFW handle to avoid pulling OpenGL headers into every TU
struct GLFWwindow;

class ThresholdModule;
class EdgeDetectionModule;

enum class ActiveModule { None, Threshold, EdgeDetection };

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

    std::unique_ptr<ThresholdModule>    m_thresholdModule;
    std::unique_ptr<EdgeDetectionModule> m_edgeModule;
};
