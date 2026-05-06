# CV Suite — OpenCV + ImGui

Aplicación de visión por computadora en C++ con interfaz gráfica.

## Funcionalidades

- Thresholding: Otsu, Adaptivo, Segmentación médica e industrial
- Detección de bordes: Canny, extracción de contornos

## Tecnologías

C++ 17 · OpenCV 4 · ImGui · GLFW · OpenGL 3.3 · CMake · vcpkg

---

## Windows — Ejecutar sin instalar nada

Descarga el archivo `CVSuite-windows.zip` de la sección [Releases](../../releases) de este repositorio, descomprímelo y ejecuta `CVSuite.exe`. No se requiere ninguna instalación adicional.

---

## Windows — Compilar desde el código fuente

### Requisitos

- **Visual Studio 2022** con el workload **"Desarrollo de escritorio con C++"** (incluye MSVC, CMake y el SDK de Windows)
- Git

### Pasos

1. Clona el repositorio:
   ```
   git clone --recurse-submodules <URL-del-repo>
   cd cv-suite-opencv-imgui-version_MacOs-
   ```

2. Ejecuta el script de bootstrap (instala dependencias, configura y compila):
   ```
   bootstrap.bat
   ```

3. El ejecutable queda en `build\bin\Release\CVSuite.exe` con todas las DLLs necesarias al lado. Se puede copiar y ejecutar en cualquier PC con Windows sin instalar nada más.

> El primer build descarga las dependencias automáticamente via vcpkg y puede tardar unos minutos.

---

## macOS — Compilar desde el código fuente

### Requisitos

- Xcode Command Line Tools
- [Homebrew](https://brew.sh): `brew install opencv glfw`
- CMake: `brew install cmake`

### Pasos

```bash
cmake -B build
cmake --build build
./build/bin/CVSuite
```

---

## Estructura del proyecto

```
bootstrap.bat          # Script de setup automático (Windows)
CMakeLists.txt         # Configuración de build multiplataforma
vcpkg.json             # Dependencias gestionadas por vcpkg
main.cpp               # Punto de entrada
app/                   # Ventana principal, loop de render, navegación
core/                  # Carga de imágenes y conversión Mat→Textura OpenGL
modules/               # Módulos de procesamiento (Threshold, EdgeDetection)
vcpkg/                 # Gestor de paquetes C++ (incluido en el repo)
```
