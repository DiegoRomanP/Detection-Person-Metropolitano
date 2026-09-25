# =============================================================
# CONTEO DE PERSONAS - basado en el script del profesor
# =============================================================
# Idea: el script de clase (professor_scripts/main.py) detecta
# "cambio de textura" en una celda fija con:
#     gris -> adaptiveThreshold -> medianBlur -> dilate -> countNonZero
# Eso sirve para saber si UNA celda fija está ocupada (parqueo).
#
# Aquí el objetivo es distinto: contar personas que ATRAVIESAN una
# región (no una celda fija), así que hace falta saber además QUÉ
# parte de la imagen se movió. Por eso se agrega un sustractor de
# fondo (MOG2) antes del mismo pipeline de limpieza del profesor,
# y un tracking simple por cercanía de centroides para no contar
# dos veces a la misma persona mientras cruza el ROI.
#
# La estabilización de video se deja pendiente a propósito.
# =============================================================
import cv2
import numpy as np

# =============================================================
# CAPTURA DE VIDEO
# =============================================================
video = cv2.VideoCapture("video_grafica.mp4")

# =============================================================
# REGIÓN DE INTERÉS (ROI)
# El video es vertical (478x850). Se usa una franja horizontal
# sobre la rampa peatonal, cerca de la parte baja de la imagen
# donde el flujo de personas es más claro.
# Ajustar con selecto_ROI.py si la zona no es la correcta.
# =============================================================
x0, y0, x1, y1 = 0, 480, 478, 795

# =============================================================
# SUSTRACTOR DE FONDO
# Único agregado real frente al script de clase: sin esto no se
# puede distinguir "persona moviéndose" de "piso con textura".
# =============================================================
bg = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40,
    detectShadows=True,
)

# =============================================================
# KERNEL (igual idea que el kernel del profesor: np.ones((5,5)))
# =============================================================
kernel = np.ones((5, 5), np.uint8)

# =============================================================
# FILTROS DE TAMAÑO PARA DESCARTAR RUIDO
# =============================================================
AREA_MIN = 150
AREA_MAX = 15000

# =============================================================
# TRACKING SIMPLE (solo lo necesario para no contar 2 veces)
# =============================================================
tracks = {}          # {id: [cx, cy, frames_perdido]}
next_id = 0
total_personas = 0   # contador acumulado (personas distintas que cruzaron el ROI)

DIST_MAX = 50         # distancia máx. (px) para considerar que es la misma persona
FRAMES_PERDIDO_MAX = 20  # frames sin verla antes de darla por salida del ROI

while True:
    check, img = video.read()
    if not check:
        break

    # ---------------------------------------------------------
    # 1) RECORTE DE LA ROI
    # ---------------------------------------------------------
    roi = img[y0:y1, x0:x1]

    # ---------------------------------------------------------
    # 2) AISLAR EL MOVIMIENTO (lo que agrega este script)
    # ---------------------------------------------------------
    fg = bg.apply(roi)
    fg[fg == 127] = 0  # descartar sombras (detectShadows=True las marca en 127)

    # ---------------------------------------------------------
    # 3) MISMO PIPELINE DE LIMPIEZA QUE EL SCRIPT DEL PROFESOR
    #    (adaptiveThreshold -> medianBlur -> dilate)
    #    Aquí adaptiveThreshold se aplica sobre la máscara de
    #    movimiento en vez de sobre la imagen en gris directa,
    #    porque ya estamos partiendo de "lo que se movió".
    # ---------------------------------------------------------
    fgTH = cv2.adaptiveThreshold(
        fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -16
    )
    fgMedian = cv2.medianBlur(fgTH, 5)
    fgDil = cv2.dilate(fgMedian, kernel)

    # ---------------------------------------------------------
    # 4) CONTORNOS -> CENTROIDES (en vez de countNonZero por
    #    celda fija, porque aquí las personas se mueven)
    # ---------------------------------------------------------
    contornos, _ = cv2.findContours(fgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < AREA_MIN or area > AREA_MAX:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        centros.append((cx, cy))

        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ---------------------------------------------------------
    # 5) ENVEJECER TRACKS EXISTENTES
    # ---------------------------------------------------------
    for tid in tracks:
        tracks[tid][2] += 1

    # ---------------------------------------------------------
    # 6) ASIGNAR CENTROIDES A TRACKS (par más cercano primero)
    #    Antes: cada centroide buscaba su track más cercano de
    #    forma independiente, sin marcar el track como "usado".
    #    Si dos personas distintas caían cerca del mismo track,
    #    ambas lo reclamaban y una terminaba sobrescribiendo a la
    #    otra sin crear un track nuevo -> se perdía gente ya
    #    detectada. Ahora se arma la lista de todos los pares
    #    (centroide, track) posibles y se asignan en orden de
    #    distancia, así un track ya asignado no puede ser
    #    reclamado por un segundo centroide en el mismo frame.
    # ---------------------------------------------------------
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
        total_personas += 1  # persona nueva detectada en el ROI

    # ---------------------------------------------------------
    # 7) ELIMINAR TRACKS QUE YA SALIERON DEL ROI
    # ---------------------------------------------------------
    for tid in list(tracks.keys()):
        if tracks[tid][2] > FRAMES_PERDIDO_MAX:
            del tracks[tid]

    # ---------------------------------------------------------
    # 8) DIBUJAR ROI Y CONTEOS
    # ---------------------------------------------------------
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(img, f"En el ROI ahora: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, f"Total contado: {total_personas}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("video", img)
    cv2.imshow("mascara", fgDil)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC para salir
        break

print(f"Total de personas contadas: {total_personas}")

video.release()
cv2.destroyAllWindows()
