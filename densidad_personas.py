# =============================================================
# PROTOTIPO: ESTIMACIÓN POR DENSIDAD DE CUADRÍCULA (GRID)
# =============================================================
# Idea (propuesta por el usuario): en vez de detectar contornos y
# rastrear centroides, se divide el ROI en celdas pequeñas. Una
# celda se marca "activada" si tiene movimiento (foreground). El
# número de personas EN ESE INSTANTE se estima como:
#
#       personas_estimadas = celdas_activadas / CELDAS_POR_PERSONA
#
# Esto es una estimación de DENSIDAD (cuántas personas hay ahora
# en el ROI), no un conteo acumulado de cuántas personas distintas
# cruzaron el lugar durante el video. Por eso este script muestra
# también, a modo de comparación, el conteo acumulado del tracker
# de contador_personas.py.
#
# CELDAS_POR_PERSONA se ajustó viendo la cuadrícula dibujada sobre
# el video: una persona activa ~8 celdas de 20x20 px. El valor
# calculado a partir del área mediana de un contorno de persona
# (~690 px^2 / 400 px^2 por celda) daba ~2, pero eso subestima
# mucho lo que realmente se ve activado en pantalla (ver también
# la nota sobre perspectiva/oclusión: 8 es más representativo del
# centro del ROI, pero seguirá variando según la distancia a la
# cámara).
# =============================================================
import cv2
import numpy as np

# =============================================================
# CAPTURA DE VIDEO
# =============================================================
video = cv2.VideoCapture("video_grafica.mp4")

# =============================================================
# REGIÓN DE INTERÉS (ROI)
# Más alta que en contador_personas.py, para cubrir mejor la
# rampa por la que camina la gente.
# =============================================================
x0, y0, x1, y1 = 0, 480, 380, 795

# =============================================================
# SUSTRACTOR DE FONDO + LIMPIEZA (igual que contador_personas.py)
# =============================================================
bg = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40,
    detectShadows=True,
)
kernel = np.ones((5, 5), np.uint8)
kernel_ruido = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# =============================================================
# PARÁMETROS DE LA CUADRÍCULA
# =============================================================
CELL = 20                  # tamaño de celda en píxeles (cuadrada)
UMBRAL_ACTIVACION = 0.35   # % de píxeles blancos en la celda para considerarla "activada"
CELDAS_POR_PERSONA = 8     # ajustado a ojo viendo la cuadrícula sobre el video

# =============================================================
# FILTROS DE TAMAÑO PARA EL TRACKER DE COMPARACIÓN
# =============================================================
AREA_MIN = 150
AREA_MAX = 15000

# =============================================================
# TRACKING SIMPLE (idéntico a contador_personas.py, solo para
# poder mostrar el conteo acumulado de referencia)
# =============================================================
tracks = {}
next_id = 0
total_personas = 0
DIST_MAX = 50
FRAMES_PERDIDO_MAX = 20

while True:
    check, img = video.read()
    if not check:
        break

    # ---------------------------------------------------------
    # 1) RECORTE DE LA ROI
    # ---------------------------------------------------------
    roi = img[y0:y1, x0:x1]
    h_roi, w_roi = roi.shape[:2]

    # ---------------------------------------------------------
    # 2) MISMO PIPELINE DE DETECCIÓN DE MOVIMIENTO
    # ---------------------------------------------------------
    fg = bg.apply(roi)
    fg[fg == 127] = 0
    fgTH = cv2.adaptiveThreshold(
        fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -16
    )
    fgMedian = cv2.medianBlur(fgTH, 5)
    fgDil = cv2.dilate(fgMedian, kernel)

    # Quitar estructuras delgadas (p.ej. la línea amarilla) que
    # activarían celdas sin que haya una persona real: una línea
    # fina no sobrevive una apertura con un kernel más grande,
    # a diferencia de un contorno de persona.
    fgDil = cv2.morphologyEx(fgDil, cv2.MORPH_OPEN, kernel_ruido)

    # ---------------------------------------------------------
    # 3) CUADRÍCULA: RECORRER CELDAS Y CONTAR ACTIVADAS
    # ---------------------------------------------------------
    celdas_activadas = 0
    for cy in range(0, h_roi - CELL + 1, CELL):
        for cx in range(0, w_roi - CELL + 1, CELL):
            celda = fgDil[cy:cy + CELL, cx:cx + CELL]
            ocupacion = cv2.countNonZero(celda) / float(CELL * CELL)

            if ocupacion >= UMBRAL_ACTIVACION:
                celdas_activadas += 1
                cv2.rectangle(roi, (cx, cy), (cx + CELL, cy + CELL), (0, 200, 255), 1)
            else:
                cv2.rectangle(roi, (cx, cy), (cx + CELL, cy + CELL), (60, 60, 60), 1)

    personas_densidad = celdas_activadas / CELDAS_POR_PERSONA

    # ---------------------------------------------------------
    # 4) CONTORNOS + TRACKING (solo para comparar con el método
    #    de contador_personas.py)
    # ---------------------------------------------------------
    contornos, _ = cv2.findContours(fgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < AREA_MIN or area > AREA_MAX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        centros.append((x + w // 2, y + h // 2))

    for tid in tracks:
        tracks[tid][2] += 1

    # Asignación exclusiva por par más cercano (ver comentario en
    # contador_personas.py): evita que dos centroides distintos
    # reclamen el mismo track y uno se pierda sin generar uno nuevo.
    pares = []
    for ci, (cx, cy) in enumerate(centros):
        for tid, (tx, ty, _) in tracks.items():
            dist = float(np.hypot(cx - tx, cy - ty))
            if dist < DIST_MAX:
                pares.append((dist, ci, tid))
    pares.sort(key=lambda p: p[0])

    centro_usado = [False] * len(centros)
    track_usado = set()

    for dist, ci, tid in pares:
        if centro_usado[ci] or tid in track_usado:
            continue
        cx, cy = centros[ci]
        tracks[tid] = [cx, cy, 0]
        centro_usado[ci] = True
        track_usado.add(tid)

    for ci, (cx, cy) in enumerate(centros):
        if centro_usado[ci]:
            continue
        tracks[next_id] = [cx, cy, 0]
        next_id += 1
        total_personas += 1

    for tid in list(tracks.keys()):
        if tracks[tid][2] > FRAMES_PERDIDO_MAX:
            del tracks[tid]

    # ---------------------------------------------------------
    # 5) DIBUJAR ROI Y COMPARATIVA DE MÉTODOS
    # ---------------------------------------------------------
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(img, f"Densidad (grid): {personas_densidad:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(img, f"Tracking ahora: {len(tracks)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"Tracking total: {total_personas}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("video", img)
    cv2.imshow("mascara + grid", fgDil)

    if cv2.waitKey(30) & 0xFF == 27:
        break

print(f"Total tracking (referencia): {total_personas}")

video.release()
cv2.destroyAllWindows()
