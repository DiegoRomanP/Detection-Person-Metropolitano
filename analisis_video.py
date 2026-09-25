"""Analiza candidatos a personas oscuras dentro de una ROI de un video.

Tecnicas preseleccionadas de las semanas 1 a 4:
- Captura de video y seleccion de ROI.
- Conversion a escala de grises y filtro gaussiano.
- Umbralizacion Otsu inversa y dilatacion.
- Componentes conectados y propiedades de regiones: area, excentricidad y bbox.

No usa sustraccion de fondo MOG2, contornos, fusion de cajas ni tracking.

Uso:
    python analisis_video.py ruta/al/video.mp4
"""

from pathlib import Path
import sys

import cv2
import numpy as np
from skimage.measure import label, regionprops


MIN_AREA = 80
MAX_AREA = 8500
MIN_ANCHO = 5
MAX_ANCHO = 50
MIN_ALTO = 10
MAX_ALTO = 170
MIN_ASPECTO = 0.7
MAX_ASPECTO = 8.0
MIN_EXCENTRICIDAD = 0.3


def seleccionar_roi(ruta_video: Path) -> tuple[int, int, int, int] | None:
    captura = cv2.VideoCapture(str(ruta_video))
    leido, primer_frame = captura.read()
    captura.release()

    if not leido:
        print(f"No se pudo leer el primer frame de: {ruta_video}")
        return None

    roi = cv2.selectROI(
        "Selecciona la zona de analisis y presiona ENTER",
        primer_frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()

    x, y, ancho, alto = (int(valor) for valor in roi)
    if ancho == 0 or alto == 0:
        print("No se selecciono una zona de analisis.")
        return None

    return x, y, ancho, alto


def obtener_candidatos(mascara: np.ndarray) -> list[tuple[int, int, int, int]]:
    etiquetas = label(mascara > 0)
    candidatos = []

    for region in regionprops(etiquetas):
        min_fila, min_columna, max_fila, max_columna = region.bbox
        ancho = max_columna - min_columna
        alto = max_fila - min_fila
        aspecto = alto / ancho if ancho else 0.0

        if not MIN_AREA <= region.area <= MAX_AREA:
            continue
        if not MIN_ANCHO <= ancho <= MAX_ANCHO:
            continue
        if not MIN_ALTO <= alto <= MAX_ALTO:
            continue
        if not MIN_ASPECTO <= aspecto <= MAX_ASPECTO:
            continue
        if region.eccentricity < MIN_EXCENTRICIDAD:
            continue

        candidatos.append((min_columna, min_fila, ancho, alto))

    return candidatos


def analizar_video(ruta_video: Path, roi: tuple[int, int, int, int]) -> None:
    captura = cv2.VideoCapture(str(ruta_video))
    if not captura.isOpened():
        print(f"No se pudo abrir el video: {ruta_video}")
        return

    x_roi, y_roi, ancho_roi, alto_roi = roi
    kernel = np.ones((3, 3), np.uint8)

    while True:
        leido, frame = captura.read()
        if not leido:
            break

        zona = frame[y_roi:y_roi + alto_roi, x_roi:x_roi + ancho_roi]
        gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        gris_suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
        _, mascara = cv2.threshold(
            gris_suavizado,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        mascara = cv2.dilate(mascara, kernel)

        candidatos = obtener_candidatos(mascara)
        for x, y, ancho, alto in candidatos:
            cv2.rectangle(
                frame,
                (x_roi + x, y_roi + y),
                (x_roi + x + ancho, y_roi + y + alto),
                (0, 255, 0),
                2,
            )

        cv2.rectangle(
            frame,
            (x_roi, y_roi),
            (x_roi + ancho_roi, y_roi + alto_roi),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Personas estimadas: {len(candidatos)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Analisis de video", frame)
        cv2.imshow("Mascara Otsu", mascara)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    captura.release()
    cv2.destroyAllWindows()


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python analisis_video.py ruta/al/video.mp4")
        return

    ruta_video = Path(sys.argv[1]).expanduser()
    if not ruta_video.is_file():
        print(f"No existe el video: {ruta_video}")
        return

    roi = seleccionar_roi(ruta_video)
    if roi is not None:
        analizar_video(ruta_video, roi)


if __name__ == "__main__":
    main()
