# =============================================================
# IMPORTACIONES
# =============================================================
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np


# =============================================================
# TIPOS AUXILIARES
# =============================================================
Punto = Tuple[float, float]                     # (x, y) en coordenadas del frame original
Track = List[float]                             # [cx, cy, missing]
Mascara = np.ndarray                            # Imagen binaria uint8
Imagen = np.ndarray                             # Imagen BGR uint8


# =============================================================
# CAPTURA DE VIDEO
# =============================================================
cap: cv2.VideoCapture = cv2.VideoCapture("video_grafica.mp4")


# =============================================================
# BACKGROUND SUBTRACTOR (MOG2)
# =============================================================
bg: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
    history=100,        # Frames usados para aprender el fondo.
    varThreshold=40,    # Umbral de Mahalanobis: sensibilidad al cambio.
    detectShadows=True, # Marca sombras con 127 en vez de 255.
)


# =============================================================
# KERNEL MORFOLÓGICO
# =============================================================
kernel: Mascara = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Kernel elíptico 3x3 usado en open/dilate.


# =============================================================
# ESTRUCTURAS DE TRACKING
# =============================================================
tracks: Dict[int, Track] = {}                       # {id: [cx, cy, missing]}
historiales: Dict[int, deque[Punto]] = {}           # {id: deque de (cx, cy)}
next_id: int = 0                                    # Contador de IDs.

# --- Parámetros de tracking ---
max_dist: int = 50          # Distancia máx. para asociar centroide a track.
max_missing: int = 30       # Frames sin ver un track antes de eliminarlo.
k_frames: int = 15           # Cuántos frames atrás para calcular dirección.


# =============================================================
# REGIÓN DE INTERÉS (ROI)
# =============================================================
x0: int = 3
y0: int = 193
x1: int = 845
y1: int = 300
ROI: Tuple[int, int, int, int] = (x0, y0, x1, y1)


# =============================================================
# ESCALADO
# =============================================================
ESCALA: float = 3.0
INTERP_UP: int = cv2.INTER_CUBIC
INTERP_DOWN: int = cv2.INTER_AREA


# =============================================================
# ÁREAS Y FORMA
# =============================================================
min_area: int = 100                 # Área mínima (px del frame original).
max_area: int = 20000               # Área máxima (px del frame original).
min_area_scaled: float = min_area * (ESCALA ** 2)
max_area_scaled: float = max_area * (ESCALA ** 2)

# Relación de aspecto (alto / ancho) para descartar objetos que no son personas.
ASPECTO_MIN: float = 1.0
ASPECTO_MAX: float = 6.0


# =============================================================
# UMBRAL DE OSCURIDAD
# =============================================================
UMBRAL_OSCURO: int = 50
SOLO_OSCURO: bool = True


# =============================================================
# FUNCIÓN: máscara de píxeles oscuros
# =============================================================
def mascara_oscura(bgr: Imagen) -> Mascara:
    """Devuelve máscara binaria 255 donde el pixel es oscuro (gray <= UMBRAL_OSCURO)."""
    gray: Mascara = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, UMBRAL_OSCURO, 255, cv2.THRESH_BINARY_INV)
    # THRESH_BINARY_INV: oscuros -> 255, claros -> 0.
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


# =============================================================
# FUNCIÓN: escalar imagen
# =============================================================
def escalar(img: Imagen, escala: float) -> Imagen:
    """Redimensiona una imagen según el factor de escala."""
    if escala == 1.0:
        return img
    interp: int = INTERP_UP if escala > 1.0 else INTERP_DOWN
    return cv2.resize(img, None, fx=escala, fy=escala, interpolation=interp)


# =============================================================
# LOOP PRINCIPAL
# =============================================================
while True:
    ret: bool
    frame: Imagen
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------------------------------------------
    # 1) RECORTE DE LA ROI (con clamping)
    # ---------------------------------------------------------
    rx0: int = max(0, x0)
    ry0: int = max(0, y0)
    rx1: int = min(x1, frame.shape[1])
    ry1: int = min(y1, frame.shape[0])
    roi: Imagen = frame[ry0:ry1, rx0:rx1].copy()

    # ---------------------------------------------------------
    # 2) ESCALADO DE LA ROI
    # ---------------------------------------------------------
    roi_proc: Imagen = escalar(roi, ESCALA)

    # ---------------------------------------------------------
    # 3) MÁSCARA DE OSCUROS
    # ---------------------------------------------------------
    oscuro: Mascara = mascara_oscura(roi_proc)

    # ---------------------------------------------------------
    # 4) FILTRAR EL FRAME QUE ENTRA AL SUBTRACTOR
    # ---------------------------------------------------------
    roi_filtrado: Imagen = roi_proc.copy()
    if SOLO_OSCURO:
        roi_filtrado[oscuro == 0] = 0     # Deja solo oscuros.
    else:
        roi_filtrado[oscuro > 0] = 0      # Quita oscuros.

    # ---------------------------------------------------------
    # 5) BACKGROUND SUBTRACTION
    # ---------------------------------------------------------
    fg: Mascara = bg.apply(roi_filtrado)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    # ---------------------------------------------------------
    # 6) FILTRO DE OSCURO SOBRE EL FOREGROUND
    # ---------------------------------------------------------
    if SOLO_OSCURO:
        fg[oscuro == 0] = 0
    else:
        fg[oscuro > 0] = 0

    # ---------------------------------------------------------
    # 7) LIMPIEZA MORFOLÓGICA
    # ---------------------------------------------------------
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.dilate(fg, kernel, iterations=2)

    # ---------------------------------------------------------
    # 8) DETECCIÓN DE CONTORNOS + FILTROS DE ÁREA Y FORMA
    # ---------------------------------------------------------
    contours: Tuple[List[np.ndarray], np.ndarray]
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros: List[Punto] = []
    for c in contours:
        area: float = cv2.contourArea(c)

        # Filtro por área: mínima (ruido) y máxima (objetos grandes).
        if area < min_area_scaled or area > max_area_scaled:
            continue

        x: int
        y: int
        w: int
        h: int
        x, y, w, h = cv2.boundingRect(c)

        # Filtro por relación de aspecto (alto / ancho).
        aspecto: float = h / float(w) if w > 0 else 0.0
        if aspecto < ASPECTO_MIN or aspecto > ASPECTO_MAX:
            continue

        # Conversión a coordenadas del frame original.
        cx: int = int(rx0 + (x + w / 2) / ESCALA)
        cy: int = int(ry0 + (y + h / 2) / ESCALA)
        bx: int = int(rx0 + x / ESCALA)
        by: int = int(ry0 + y / ESCALA)
        bw: int = int(w / ESCALA)
        bh: int = int(h / ESCALA)

        centros.append((cx, cy))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    # ---------------------------------------------------------
    # 9) ENVEJECER TRACKS
    # ---------------------------------------------------------
    for tid in tracks:
        tracks[tid][2] += 1

    # ---------------------------------------------------------
    # 10) ASIGNACIÓN GREEDY DE CENTROIDES A TRACKS
    # ---------------------------------------------------------
    cen: Punto
    for cen in centros:
        mejor_id: int | None = None
        mejor_dist: float = float(max_dist)

        for tid, (tx, ty, _) in tracks.items():
            d: float = float(np.hypot(cen[0] - tx, cen[1] - ty))
            if d < mejor_dist:
                mejor_dist = d
                mejor_id = tid

        if mejor_id is not None:
            tracks[mejor_id][0] = cen[0]
            tracks[mejor_id][1] = cen[1]
            tracks[mejor_id][2] = 0
            historiales[mejor_id].append(cen)
        else:
            tracks[next_id] = [cen[0], cen[1], 0]
            historiales[next_id] = deque(maxlen=30)
            historiales[next_id].append(cen)
            next_id += 1

    # ---------------------------------------------------------
    # 11) ELIMINAR TRACKS PERDIDOS
    # ---------------------------------------------------------
    for tid in list(tracks.keys()):
        if tracks[tid][2] > max_missing:
            del tracks[tid]
            del historiales[tid]

    # ---------------------------------------------------------
    # 12) CÁLCULO Y DIBUJO DE DIRECCIÓN
    # ---------------------------------------------------------
    for tid, (cx_f, cy_f, missing) in tracks.items():
        hist: deque[Punto] = historiales[tid]
        if len(hist) > k_frames:
            x_prev: float
            y_prev: float
            x_prev, y_prev = hist[-k_frames - 1]

            dx: float = cx_f - x_prev
            dy: float = cy_f - y_prev

            if abs(dx) > 1 or abs(dy) > 1:
                ang: float = math.degrees(math.atan2(dy, dx))
                fin_x: int = int(cx_f + dx * 4)
                fin_y: int = int(cy_f + dy * 4)

                cv2.arrowedLine(
                    frame, (int(cx_f), int(cy_f)), (fin_x, fin_y),
                    (255, 0, 0), 2,
                )
                cv2.putText(
                    frame,
                    f"{ang:.0f}°",
                    (int(cx_f) + 5, int(cy_f) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

    # ---------------------------------------------------------
    # 13) DIBUJAR ROI Y CONTEO
    # ---------------------------------------------------------
    cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 255), 1)
    cv2.putText(
        frame,
        f"Tracks: {len(tracks)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # ---------------------------------------------------------
    # 14) VISUALIZACIÓN
    # ---------------------------------------------------------
    cv2.imshow("Conteo y dirección", frame)
    cv2.imshow("Mascara", fg)
    cv2.imshow("Oscuros detectados", oscuro)
    if ESCALA > 1.0:
        cv2.imshow("ROI ampliada", roi_proc)

    # ---------------------------------------------------------
    # 15) SALIR CON ESC
    # ---------------------------------------------------------
    if cv2.waitKey(30) & 0xFF == 27:
        break


# =============================================================
# LIBERAR RECURSOS
# =============================================================
cap.release()
cv2.destroyAllWindows()
