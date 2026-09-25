# =============================================================
# JUSTIFICACIÓN: por qué no basta con el umbral del profesor
# =============================================================
# El script de clase (professor_scripts/main.py) usa, sobre una
# celda de parqueo FIJA y una cámara FIJA:
#     gris -> adaptiveThreshold -> medianBlur -> dilate -> countNonZero
# Esa técnica detecta "cambio de textura local" en una imagen fija:
# funciona para saber si un auto (con su textura/sombra) está sobre
# una celda, comparado contra el asfalto liso de fondo.
#
# Este script aplica EXACTAMENTE esa misma técnica (mismos parámetros
# que professor_scripts/main.py) sobre todo el frame de
# video_grafica.mp4, sin ningún agregado, para mostrar en video que
# NO sirve para aislar personas moviéndose: se activa con cualquier
# textura de la escena (piso, rejas, paredes), esté alguien ahí o no.
#
# Al lado se muestra qué detecta un sustractor de fondo (MOG2), que
# sí distingue "esto cambió respecto al fondo aprendido" de "esto es
# textura fija" -> por eso fue necesario agregarlo en
# contador_personas.py / densidad_personas.py / densidad_por_bandas.py.
#
# Para que la comparación sea justa, ambas técnicas ven el mismo
# frame de entrada (nada de recortes de ROI ni limpieza extra).
# =============================================================
import cv2
import numpy as np

video = cv2.VideoCapture("video_grafica.mp4")

# --- Sustractor de fondo (para el lado "con MOG2") ---
bg = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40,
    detectShadows=True,
)

# --- Kernel, igual que en professor_scripts/main.py ---
kernel = np.ones((5, 5), np.uint8)

while True:
    check, frame = video.read()
    if not check:
        break

    # ===========================================================
    # TÉCNICA DEL PROFESOR, SIN CAMBIOS (ver professor_scripts/main.py)
    # ===========================================================
    imgBN = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(
        imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    imgMedian = cv2.medianBlur(imgTH, 5)
    imgDil = cv2.dilate(imgMedian, kernel)

    # ===========================================================
    # TÉCNICA CON MOG2 (lo que sí aísla movimiento)
    # ===========================================================
    fg = bg.apply(frame)
    fg[fg == 127] = 0  # descartar sombras

    # ===========================================================
    # MEDIR QUÉ TANTO DE LA IMAGEN "SE ACTIVA" EN CADA CASO
    # ===========================================================
    total_px = frame.shape[0] * frame.shape[1]
    pct_ingenuo = 100.0 * cv2.countNonZero(imgDil) / total_px
    pct_mog = 100.0 * cv2.countNonZero(fg) / total_px

    # ===========================================================
    # ETIQUETAS Y VISUALIZACIÓN
    # ===========================================================
    vista_ingenua = cv2.cvtColor(imgDil, cv2.COLOR_GRAY2BGR)
    cv2.putText(vista_ingenua, f"Umbral del profesor: {pct_ingenuo:.1f}% del frame activado",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vista_ingenua, "(se activa con piso, rejas, paredes... no solo personas)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    vista_mog = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    cv2.putText(vista_mog, f"MOG2 (fondo aprendido): {pct_mog:.1f}% del frame activado",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vista_mog, "(solo lo que se mueve respecto al fondo)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Original", frame)
    cv2.imshow("Umbral del profesor aplicado directo (NO aisla personas)", vista_ingenua)
    cv2.imshow("Con sustraccion de fondo MOG2 (SI aisla movimiento)", vista_mog)

    if cv2.waitKey(30) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
