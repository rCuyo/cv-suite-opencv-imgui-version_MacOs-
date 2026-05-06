@echo off
setlocal EnableDelayedExpansion

echo.
echo ===================================
echo   CVSuite - Bootstrap  (Windows)
echo ===================================
echo.

REM ── Buscar CMake ────────────────────────────────────────────────────────────
set CMAKE_EXE=
where cmake >nul 2>&1 && set CMAKE_EXE=cmake

if "!CMAKE_EXE!"=="" (
    for %%E in (Community Enterprise Professional BuildTools) do (
        set TRY="%ProgramFiles%\Microsoft Visual Studio\2022\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if exist !TRY! ( set CMAKE_EXE=!TRY! & goto :cmake_found )
    )
)
if "!CMAKE_EXE!"=="" (
    set TRY="%ProgramFiles%\CMake\bin\cmake.exe"
    if exist !TRY! ( set CMAKE_EXE=!TRY! & goto :cmake_found )
)

echo [ERROR] CMake no encontrado.
echo         Instala Visual Studio 2022 con el workload "Desarrollo de escritorio con C++"
echo         o descarga CMake desde https://cmake.org/download/
pause
exit /b 1

:cmake_found
echo [OK] CMake: !CMAKE_EXE!

REM ── Verificar / compilar vcpkg ───────────────────────────────────────────────
if not exist "vcpkg\vcpkg.exe" (
    echo [INFO] Compilando vcpkg por primera vez...
    call vcpkg\bootstrap-vcpkg.bat -disableMetrics
    if errorlevel 1 ( echo [ERROR] Fallo bootstrap de vcpkg. & pause & exit /b 1 )
)
echo [OK] vcpkg listo.

REM ── Configurar ───────────────────────────────────────────────────────────────
echo.
echo [1/2] Configurando proyecto con CMake...
!CMAKE_EXE! -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
if errorlevel 1 ( echo [ERROR] Fallo la configuracion de CMake. & pause & exit /b 1 )

REM ── Compilar ─────────────────────────────────────────────────────────────────
echo.
echo [2/2] Compilando...
!CMAKE_EXE! --build build --config Release
if errorlevel 1 ( echo [ERROR] Fallo la compilacion. & pause & exit /b 1 )

echo.
echo ===================================
echo  Build completado exitosamente.
echo  Ejecutable: build\bin\Release\CVSuite.exe
echo ===================================
echo.
pause
