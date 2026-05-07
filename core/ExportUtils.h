#pragma once
#include <string>

// Genera la siguiente ruta disponible con numeración incremental.
// Ejemplo: nextExportPath("exports", "ocr_resultado", "png")
//   → "exports/ocr_resultado_1.png" si no existe ninguno
//   → "exports/ocr_resultado_3.png" si ya existen _1 y _2
//
// Si dir está vacío, los archivos se crean en el directorio de trabajo actual.
namespace ExportUtils
{
    std::string nextExportPath(const std::string& dir,
                               const std::string& prefix,
                               const std::string& ext = "png");
}
