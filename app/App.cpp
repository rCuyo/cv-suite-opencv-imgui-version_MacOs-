// OpenGL 3.3 core profile — debe incluirse ANTES de GLFW cuando GLFW_INCLUDE_NONE está activo
#include "core/gl.h"
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "App.h"
#include "modules/ThresholdModule.h"
#include "modules/EdgeDetectionModule.h"
#include "modules/DocumentCleanupModule.h"
#include "modules/TransformModule.h"
#include "modules/HistogramEqualizationModule.h"
#include "modules/NoiseReductionModule.h"
#include "modules/MorphologyModule.h"
#include "modules/HOGPedestrianModule.h"

#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

static void glfwErrorCallback(int error, const char* description)
{
    std::fprintf(stderr, "[GLFW] Error %d: %s\n", error, description);
}

// ─────────────────────────────────────────────────────────────────────────────

App::App(const std::string& title, int width, int height)
    : m_title(title), m_width(width), m_height(height)
{}

App::~App()
{
    shutdown();
}

// ─── init ────────────────────────────────────────────────────────────────────

bool App::init()
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        std::fprintf(stderr, "[GLFW] glfwInit() failed.\n");
        return false;
    }

    // Request OpenGL 3.3 Core Profile (required on macOS)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Mandatory on macOS

    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::fprintf(stderr, "[GLFW] Failed to create window.\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);

#ifdef _WIN32
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "[GLAD] No se pudo inicializar el cargador de OpenGL.\n");
        glfwTerminate();
        return false;
    }
#endif

    glfwSwapInterval(1); // vsync

    // ── ImGui ────────────────────────────────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Latin-1 Supplement cubre caracteres españoles: á é í ó ú ñ ü ¡ ¿
    static const ImWchar latinRanges[] = { 0x0020, 0x00FF, 0 };
#ifdef _WIN32
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 15.0f, nullptr, latinRanges);
#elif defined(__APPLE__)
    io.Fonts->AddFontFromFileTTF("/System/Library/Fonts/Helvetica.ttc", 15.0f, nullptr, latinRanges);
#endif

    ImGui::StyleColorsDark();

    // Tweak style for a cleaner look
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding   = 4.0f;
    style.FrameRounding    = 3.0f;
    style.GrabRounding     = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.ItemSpacing      = ImVec2(8, 6);

    // Backends: GLFW + OpenGL 3 / GLSL 150
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // ── Modules ──────────────────────────────────────────────────────────────
    m_thresholdModule  = std::make_unique<ThresholdModule>();
    m_edgeModule       = std::make_unique<EdgeDetectionModule>();
    m_docCleanupModule = std::make_unique<DocumentCleanupModule>();
    m_transformModule  = std::make_unique<TransformModule>();
    m_histEqModule     = std::make_unique<HistogramEqualizationModule>();
    m_noiseModule      = std::make_unique<NoiseReductionModule>();
    m_morphologyModule = std::make_unique<MorphologyModule>();
    m_hogModule        = std::make_unique<HOGPedestrianModule>();

    return true;
}

// ─── Main loop ───────────────────────────────────────────────────────────────

void App::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        renderUI();

        ImGui::Render();

        int fbW, fbH;
        glfwGetFramebufferSize(m_window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(m_window);
    }
}

// ─── UI ──────────────────────────────────────────────────────────────────────

void App::renderUI()
{
    // Cover the entire OS window with a single borderless ImGui window
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);

    ImGuiWindowFlags mainFlags =
        ImGuiWindowFlags_NoTitleBar  | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize    | ImGuiWindowFlags_NoMove     |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##Root", nullptr, mainFlags);
    ImGui::PopStyleVar(3);

    // ── Sidebar (fixed 210 px wide) ──────────────────────────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
    ImGui::BeginChild("##Sidebar", ImVec2(210, 0), ImGuiChildFlags_Borders);
    renderSidebar();
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::SameLine();

    // ── Content area (fills remaining space, scrollable) ─────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 10));
    ImGui::BeginChild("##Content", ImVec2(0, 0), ImGuiChildFlags_None,
                      ImGuiWindowFlags_HorizontalScrollbar);
    renderContent();
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::End();
}

void App::renderSidebar()
{
    // ── Branding ──────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "CV Suite");
    ImGui::TextDisabled("OpenCV %d.%d", CV_MAJOR_VERSION, CV_MINOR_VERSION);

    // Full-width nav button, highlighted when its module is active
    auto navButton = [&](const char* label, ActiveModule mod) {
        bool active = (m_activeModule == mod);
        if (active)
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label, ImVec2(-1, 32)))
            m_activeModule = mod;
        if (active)
            ImGui::PopStyleColor();
    };

    // Colored section divider with module number and subtitle
    auto sectionHeader = [&](const char* number, const char* subtitle, ImVec4 color) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(color, "%s", number);
        ImGui::TextDisabled("  %s", subtitle);
        ImGui::Spacing();
    };

    // ── MÓDULO 1 — Segmentación y Bordes ─────────────────────────────────────
    sectionHeader("MÓDULO 1", "Segmentación y Bordes",
                  ImVec4(0.40f, 0.85f, 1.00f, 1.0f));
    navButton("Umbralización",            ActiveModule::Threshold);
    navButton("Detección de bordes",      ActiveModule::EdgeDetection);
    navButton("Document Cleanup",         ActiveModule::DocumentCleanup);

    // ── MÓDULO 2 — Transformaciones Geométricas y Mejora de Imagen ───────────
    sectionHeader("MÓDULO 2", "Transf. y Mejora de Imagen",
                  ImVec4(0.20f, 1.00f, 0.50f, 1.0f));
    navButton("Geometric Transformations", ActiveModule::Transform);
    navButton("Histogram Equalization",    ActiveModule::HistogramEq);

    navButton("Noise Reduction & Filters",  ActiveModule::NoiseReduction);
    navButton("Morphological Operations",  ActiveModule::Morphology);

    // ── MÓDULO 3 — Feature Extraction & Detection ────────────────────────────
    sectionHeader("MÓDULO 3", "Feature Extraction & Detection",
                  ImVec4(0.35f, 1.00f, 0.75f, 1.0f));
    navButton("HoG Pedestrian Detection", ActiveModule::HOGPedestrian);

    // ── Pie de versión ────────────────────────────────────────────────────────
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 24.0f);
    ImGui::Separator();
    ImGui::TextDisabled(" v1.2");
}

void App::renderContent()
{
    switch (m_activeModule)
    {
    case ActiveModule::None:
        ImGui::Spacing();
        ImGui::SetCursorPosX(
            (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("Selecciona un módulo del panel lateral.").x) * 0.5f);
        ImGui::TextDisabled("Selecciona un módulo del panel lateral.");
        break;

    case ActiveModule::Threshold:
        m_thresholdModule->renderUI();
        break;

    case ActiveModule::EdgeDetection:
        m_edgeModule->renderUI();
        break;

    case ActiveModule::DocumentCleanup:
        m_docCleanupModule->renderUI();
        break;

    case ActiveModule::Transform:
        m_transformModule->renderUI();
        break;

    case ActiveModule::HistogramEq:
        m_histEqModule->renderUI();
        break;

    case ActiveModule::NoiseReduction:
        m_noiseModule->renderUI();
        break;

    case ActiveModule::Morphology:
        m_morphologyModule->renderUI();
        break;

    case ActiveModule::HOGPedestrian:
        m_hogModule->renderUI();
        break;
    }
}

// ─── Shutdown ────────────────────────────────────────────────────────────────

void App::shutdown()
{
    if (!m_window) return;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();
    m_window = nullptr;
}
