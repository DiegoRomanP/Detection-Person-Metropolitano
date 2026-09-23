import math
from collections import deque
from typing import Dict, List, Tuple
import cv2
import numpy as np

# =============================================================
# TIPOS AUXILIARES
# =============================================================
Punto = Tuple[float, float]
Track = List[float]
Mascara = np.ndarray
Imagen = np.ndarray

# =============================================================
# CAPTURA DE VIDEO
# =============================================================
cap = cv2.VideoCapture("video_grafica.mp4")

# =============================================================
# BACKGROUND SUBTRACTOR (MOG2)
# history bajo = se adapta más rápido (no pierde personas quietas)
# varThreshold bajo = más sensible al movimiento
# =============================================================
bg = cv2.createBackgroundSubtractorMOG2(
    history=50,
    varThreshold=30,
    detectShadows=True,
)

# =============================================================
# KERNELS MORFOLÓGICOS
# =============================================================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_union = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # Para unir fragmentos
kernel_peq = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# =============================================================
# ESTRUCTURAS DE TRACKING
# =============================================================
tracks: Dict[int, Track] = {}
historiales: Dict[int, deque[Punto]] = {}
next_id: int = 0

max_dist: int = 70       # Aumentado: las cajas fusionadas pueden saltar
max_missing: int = 30
k_frames: int = 20

# =============================================================
# ZONA DE ANÁLISIS (SOLO CENTRO)
# =============================================================
Z_X1, Z_Y1 = 60, 400
Z_X2, Z_Y2 = 350, 500

# =============================================================
# PARÁMETROS DE TAMAÑO (RELAJADOS PARA ACEPTAR FRAGMENTOS)
# =============================================================
MIN_ANCHO, MAX_ANCHO = 5, 50
MIN_ALTO,  MAX_ALTO  = 10, 170
MIN_AREA,  MAX_AREA  = 80, 8500

# =============================================================
# ESCALADO
# =============================================================
ESCALA = 3.0
INTERP_UP = cv2.INTER_CUBIC
INTERP_DOWN = cv2.INTER_AREA

# Aspecto ampliado para aceptar torsos anchos y piernas delgadas
ASPECTO_MIN = 0.2
ASPECTO_MAX = 10.0

# =============================================================
# UMBRAL DE OSCURIDAD
# =============================================================
UMBRAL_OSCURO: int = 80

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
def escalar(img: Imagen, escala: float) -> Imagen:
    if escala == 1.0:
        return img
    interp = INTERP_UP if escala > 1.0 else INTERP_DOWN
    return cv2.resize(img, None, fx=escala, fy=escala, interpolation=interp)


def mascara_oscura(bgr: Imagen) -> Mascara:
    """Devuelve máscara binaria 255 donde el pixel es oscuro (gray <= UMBRAL_OSCURO)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, UMBRAL_OSCURO, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def fusionar_cajas(boxes: List[Tuple[int, int, int, int]], 
                   dist_max: int = 40,
                   solape_min: float = 0.15) -> List[Tuple[int, int, int, int]]:
    """
    Fusiona bounding boxes cercanas o solapadas en una sola.
    boxes: lista de (x, y, w, h).
    dist_max: distancia máxima entre cajas para fusionarlas.
    solape_min: fracción mínima de solape en X o Y para considerar unión.
    """
    if len(boxes) <= 1:
        return boxes

    boxes = boxes[:]
    fusionado = True
    while fusionado:
        fusionado = False
        nuevas = []
        usados = [False] * len(boxes)

        for i in range(len(boxes)):
            if usados[i]:
                continue
            x1, y1, w1, h1 = boxes[i]
            # Centro y esquinas
            cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
            x1b, y1b = x1 + w1, y1 + h1

            for j in range(i + 1, len(boxes)):
                if usados[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
                x2b, y2b = x2 + w2, y2 + h2

                # Distancia entre centros
                dist = math.hypot(cx1 - cx2, cy1 - cy2)

                # Solape horizontal y vertical
                solape_x = max(0, min(x1b, x2b) - max(x1, x2))
                solape_y = max(0, min(y1b, y2b) - max(y1, y2))
                area_min = min(w1 * h1, w2 * h2)
                solape = (solape_x * solape_y) / area_min if area_min > 0 else 0

                # Fusionar si están cerca o si se solapan bastante
                if dist < dist_max or solape > solape_min:
                    # Caja envolvente
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1b, x2b) - nx
                    nh = max(y1b, y2b) - ny
                    nuevas.append((nx, ny, nw, nh))
                    usados[i] = True
                    usados[j] = True
                    fusionado = True
                    break
            if not usados[i]:
                nuevas.append(boxes[i])
                usados[i] = True

        boxes = nuevas

    return boxes


# =============================================================
# LOOP PRINCIPAL
# =============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) RECORTE DE LA ZONA CENTRAL
    rx0, ry0 = max(0, Z_X1), max(0, Z_Y1)
    rx1, ry1 = min(Z_X2, frame.shape[1]), min(Z_Y2, frame.shape[0])
    roi = frame[ry0:ry1, rx0:rx1].copy()

    # 2) ESCALADO DE LA ROI
    roi_proc = escalar(roi, ESCALA)

    # 3) MÁSCARA DE OSCUROS
    oscuro = mascara_oscura(roi_proc)

    # 4) FILTRAR EL FRAME: dejar solo oscuros, apagar claros
    roi_filtrado = roi_proc.copy()
    roi_filtrado[oscuro == 0] = 0

    # 5) BACKGROUND SUBTRACTION
    fg = bg.apply(roi_filtrado)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    # 6) APLICAR MÁSCARA DE OSCUROS AL FOREGROUND
    fg[oscuro == 0] = 0

    # 7) LIMPIEZA MORFOLÓGICA (CLOSE para unir fragmentos)
    #    CLOSE = dilatar y luego erosionar -> une fragmentos cercanos
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_union)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_peq)   # Quita ruido
    fg = cv2.dilate(fg, kernel_peq, iterations=1)

    # 8) DETECCIÓN DE CONTORNOS
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 9) EXTRAER CAJAS QUE PASAN LOS FILTROS
    cajas_orig: List[Tuple[int, int, int, int]] = []  # en coords originales de la zona
    for c in contours:
        area_scaled = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        area_orig = area_scaled / (ESCALA ** 2)
        w_orig = w / ESCALA
        h_orig = h / ESCALA

        # Filtros de tamaño (relajados)
        if w_orig < MIN_ANCHO or w_orig > MAX_ANCHO: continue
        if h_orig < MIN_ALTO or h_orig > MAX_ALTO: continue
        if area_orig < MIN_AREA or area_orig > MAX_AREA: continue

        # Filtro de aspecto ampliado
        aspecto = h_orig / float(w_orig) if w_orig > 0 else 0.0
        if aspecto < ASPECTO_MIN or aspecto > ASPECTO_MAX:
            continue

        # Convertir a coords originales de la zona
        bx = int(x / ESCALA)
        by = int(y / ESCALA)
        bw = int(w_orig)
        bh = int(h_orig)

        cajas_orig.append((bx, by, bw, bh))

    # 10) FUSIONAR CAJAS CERCANAS (une fragmentos de una misma persona)
    cajas_fusionadas = fusionar_cajas(cajas_orig, dist_max=50, solape_min=0.10)

    # 11) EXTRAER CENTROS DE LAS CAJAS FUSIONADAS
    centros: List[Punto] = []
    for (bx, by, bw, bh) in cajas_fusionadas:
        # Verificar que las cajas fusionadas sigan dentro de límites razonables
        w_orig = bw
        h_orig = bh
        if w_orig > MAX_ANCHO * 1.8 or h_orig > MAX_ALTO * 1.8:
            continue  # Fusión excesiva, descartar

        # Centroide global
        cx = int(rx0 + bx + bw / 2)
        cy = int(ry0 + by + bh / 2)

        # Bounding box global para dibujar
        gx = int(rx0 + bx)
        gy = int(ry0 + by)

        centros.append((cx, cy))
        cv2.rectangle(frame, (gx, gy), (gx + bw, gy + bh), (0, 255, 0), 2)

    # 12) ENVEJECER TRACKS
    for tid in tracks:
        tracks[tid][2] += 1

    # 13) ASIGNACIÓN GREEDY DE CENTROIDES A TRACKS
    for cen in centros:
        mejor_id = None
        mejor_dist = float(max_dist)

        for tid, (tx, ty, _) in tracks.items():
            d = float(np.hypot(cen[0] - tx, cen[1] - ty))
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

    # 14) ELIMINAR TRACKS PERDIDOS
    for tid in list(tracks.keys()):
        if tracks[tid][2] > max_missing:
            del tracks[tid]
            del historiales[tid]

    # 15) DIBUJAR ROI, DIRECCIÓN Y CONTEO
    cv2.rectangle(frame, (Z_X1, Z_Y1), (Z_X2, Z_Y2), (0, 255, 255), 2)

    for tid, (cx_f, cy_f, missing) in tracks.items():
        hist = historiales[tid]
        if len(hist) > k_frames:
            x_prev, y_prev = hist[-k_frames - 1]
            dx = cx_f - x_prev
            dy = cy_f - y_prev

            if abs(dx) > 2 or abs(dy) > 2:
                ang = math.degrees(math.atan2(dy, dx))
                if -45 <= ang <= 45: dir_texto = "Der"
                elif 45 < ang <= 135: dir_texto = "Abajo"
                elif -135 <= ang < -45: dir_texto = "Arriba"
                else: dir_texto = "Izq"

                fin_x = int(cx_f + dx * 4)
                fin_y = int(cy_f + dy * 4)
                cv2.arrowedLine(frame, (int(cx_f), int(cy_f)), (fin_x, fin_y), (255, 0, 0), 2)
                cv2.putText(frame, dir_texto, (int(cx_f) + 5, int(cy_f) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 16) CONTEO EN PANTALLA
    cv2.putText(frame, f"Personas (Centro): {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 17) VISUALIZACIÓN
    cv2.imshow("Analisis - Personas oscuras", frame)
    cv2.imshow("Mascara oscuros", oscuro)
    cv2.imshow("Foreground (con fusion)", fg)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()