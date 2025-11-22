"""
Asistente de insumos basado en una matriz de proximidad (phi).

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import gradio as gr


# Rutas de archivos
ORIGINAL_CSV = Path("phi.csv")
CLEAN_CSV = Path("phi_limpio.csv")


def detectar_separador(ruta: Path) -> str:
    """Detecta el separador más probable en el archivo."""
    with ruta.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                first_line = linea
                break
        else:
            return ","

    if ";" in first_line:
        return ";"
    if "\t" in first_line:
        return "\t"
    return ","


def generar_archivo_limpio(
    ruta_origen: Path = ORIGINAL_CSV,
    ruta_destino: Path = CLEAN_CSV,
) -> pd.DataFrame:
    """
    Genera archivo limpio y normalizado a partir del CSV original.

    - Detecta automáticamente el separador.
    - Interpreta comas como separador decimal.
    - Intenta convertir columnas a numérico.
    """
    if not ruta_origen.exists():
        raise FileNotFoundError(f"No se encontró el archivo de origen: {ruta_origen}")

    sep = detectar_separador(ruta_origen)

    df = pd.read_csv(
        ruta_origen,
        sep=sep,
        decimal=",",   # interpreta "0,25" como 0.25
        encoding="utf-8",
    )

    df = df.apply(pd.to_numeric, errors="ignore")
    df.to_csv(ruta_destino, index=False, encoding="utf-8")

    print(f"✅ Archivo limpio guardado como '{ruta_destino.name}'")
    return df


def cargar_matriz_proximidad(
    ruta_limpia: Path = CLEAN_CSV,
    ruta_origen: Path = ORIGINAL_CSV,
) -> pd.DataFrame:
    """Carga la matriz de proximidad entre insumos."""
    if ruta_limpia.exists():
        df = pd.read_csv(ruta_limpia, index_col=0, encoding="utf-8")
    else:
        df = generar_archivo_limpio(ruta_origen, ruta_limpia)
        # La primera columna se asume como identificador de insumo
        df = df.set_index(df.columns[0])

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="ignore")
    df = df.fillna(0)

    return df


# Cargar matriz global
phi: Optional[pd.DataFrame] = None
try:
    phi = cargar_matriz_proximidad()
except FileNotFoundError as exc:
    print(f"❌ Error al cargar la matriz de proximidad: {exc}")
    phi = None


def construir_html_compacto(df: pd.DataFrame, insumo_base: str, umbral_pct: float) -> str:
    """Construye un único bloque HTML con resumen + tabla compacta."""
    if df.empty:
        return "<p>No hay datos para mostrar.</p>"

    total = len(df)

    html = [
        f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px;">
          <p>
            <strong>Insumo base:</strong> {insumo_base}<br>
            <strong>Umbral mínimo:</strong> {umbral_pct:.1f}%<br>
            <strong>Total de insumos similares encontrados:</strong> {total}
          </p>
          <table style="border-collapse: collapse; width: 100%; font-family: monospace; font-size: 11px;">
            <thead>
              <tr>
                <th style="border: 1px solid #ccc; padding: 2px; text-align: left;">Insumo similar</th>
                <th style="border: 1px solid #ccc; padding: 2px; text-align: right;">Proximidad (%)</th>
              </tr>
            </thead>
            <tbody>
        """
    ]

    for _, fila in df.iterrows():
        prox = float(fila["Proximidad (%)"])
        if prox >= 80:
            color = "#d4edda"  # verde suave
        elif prox >= 65:
            color = "#fff3cd"  # amarillo suave
        else:
            color = "#f8d7da"  # rojo suave

        html.append(
            f"""
            <tr>
              <td style="border: 1px solid #ccc; padding: 2px;">{fila['Insumo similar']}</td>
              <td style="border: 1px solid #ccc; padding: 2px; text-align: right; background-color: {color};">
                {prox:.2f}%
              </td>
            </tr>
            """
        )

    html.append(
        """
            </tbody>
          </table>
        </div>
        """
    )

    return "\n".join(html)


def recomendar_uno_html(insumo: str, umbral_pct: float) -> str:
    """
    Recomienda insumos similares para un insumo base,
    y devuelve TODO (resumen + tabla) en un único HTML compacto.
    """
    if phi is None:
        return "<p style='color:red;'>❌ No se pudo cargar la matriz de proximidad. Verifique el archivo CSV.</p>"

    insumo = str(insumo).strip()
    if not insumo:
        return "<p>⚠️ Debe ingresar un insumo.</p>"

    if insumo not in phi.index:
        return f"<p style='color:red;'>❌ El insumo '{insumo}' no existe en la matriz.</p>"

    # Serie de proximidades (decimales 0–1) -> porcentaje
    serie = phi.loc[insumo] * 100.0

    # Filtrar por umbral
    serie = serie[serie >= umbral_pct]

    # Eliminar el propio insumo si aparece
    if insumo in serie.index:
        serie = serie.drop(insumo)

    if serie.empty:
        return (
            f"<p>⚠️ No se encontraron insumos con proximidad "
            f"≥ {umbral_pct:.1f}% para el insumo '{insumo}'.</p>"
        )

    # DataFrame ordenado
    df = serie.to_frame(name="Proximidad (%)")
    df["Insumo similar"] = df.index
    df = df[["Insumo similar", "Proximidad (%)"]]
    df["Proximidad (%)"] = df["Proximidad (%)"].round(2)
    df = df.sort_values(by="Proximidad (%)", ascending=False)

    return construir_html_compacto(df, insumo_base=insumo, umbral_pct=umbral_pct)


# Interfaz Gradio compacta
iface = gr.Interface(
    fn=recomendar_uno_html,
    inputs=[
        gr.Textbox(
            label="Insumo",
            lines=1,
            placeholder="Ejemplo: I4",
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=50,
            step=1,
            label="Umbral mínimo de proximidad (%)",
        ),
    ],
    outputs=gr.HTML(
        label="Resumen y tabla de insumos similares",
    ),
    title="🤖 Asistente de Insumos",
    description=(
        "Ingrese un insumo (por ejemplo: I4) y seleccione el umbral mínimo de proximidad "
        "en porcentaje."
    ),
)

if __name__ == "__main__":
    iface.launch()
