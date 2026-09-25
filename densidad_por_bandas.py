import cv2
import numpy as np


video = cv2.VideoCapture("video_grafica.mp4")



CAL_Y_A, CAL_CPP_A = 472, 5
CAL_Y_B, CAL_CPP_B = 687, 8


BANDAS = [
    {
        "nombre": "b0",
        "y0": 207, "y1": 311,
        "x0": 92, "x1": 252,
        "celdas_por_persona": 5,
    },
    {
        "nombre": "b1",
        "y0": 374, "y1": 542,
        "x0": 55, "x1": 229,
        "celdas_por_persona": 6,
    },
    {
        "nombre": "b2",
        "y0": 547, "y1": 829,
        "x0": 20, "x1": 283,
        "celdas_por_persona": 9,
    },
]

for b in BANDAS:
    b["bg"] = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)


kernel = np.ones((5, 5), np.uint8)
kernel_ruido = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

CELL = 20
UMBRAL_ACTIVACION = 0.35

ESTABILIZAR = True
MAX_DESPLAZAMIENTO = 5.0  # px

anterior_gris = None


def etiquetar(imagen_gris, texto):
    """Convierte una máscara en gris a BGR y le pone una etiqueta encima,
    para poder mostrar varias etapas del filtro una junto a otra."""
    bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, texto, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return bgr


while True:
    check, img = video.read()
    if not check:
        break

    # -----------------------------------------------------------
    # ESTABILIZACIÓN: alinear el frame contra el frame anterior
    # -----------------------------------------------------------
    if ESTABILIZAR:
        gray_actual = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if anterior_gris is None:
            dx, dy = 0.0, 0.0
        else:
            (dx, dy), _ = cv2.phaseCorrelate(anterior_gris, gray_actual)
            if np.hypot(dx, dy) > MAX_DESPLAZAMIENTO:
                dx, dy = 0.0, 0.0  # probable ruido por la multitud, no la cámara

        if dx != 0.0 or dy != 0.0:
            M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE)

        # el frame ya estabilizado queda como referencia del siguiente,
        # así el contenido comparado siempre es reciente y parecido.
        anterior_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    original = img.copy()
    total_densidad = 0.0
    filas_debug = []  # una fila de imágenes (etapas del filtro) por banda


    #Por cada banda
    for b in BANDAS:

        #Extraer el parche que le corresponde
        roi = img[b["y0"]:b["y1"], b["x0"]:b["x1"]]
        h_roi, w_roi = roi.shape[:2]

        #Mapear la imagen a escala de grises
        fg = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        #fg = b["bg"].apply(roi)
        #fg[fg == 127] = 0

        #Aplicar una umbralización adaptativa
        fgTH = cv2.adaptiveThreshold(
            fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -16
        )
        #Una mediana para preservar bordes
        fgMedian = cv2.medianBlur(fgTH, 5)

        # Dilate: Recorre usando un kernel la imagen. Si alguno es blanco, transforma el pixel central en blanco. Kernel todos 1 para considerar todos los pixeles de la ventana
        # morphologyEx: Preserva los píxeles que forman una forma elipsoidal (similar a una persona), elimina el resto, y luego expande de nuevo los píxeles sobrevivientes
        fgDil = cv2.dilate(fgMedian, kernel)
        fgDil = cv2.morphologyEx(fgDil, cv2.MORPH_OPEN, kernel_ruido)

        # Guardar las etapas para mostrarlas luego junto al video original
        fila_debug = np.hstack([
            etiquetar(fgTH, "2) umbral adaptativo"),
            etiquetar(fgMedian, "3) mediana"),
            etiquetar(fgDil, "4) dilatar + abrir"),
        ])
        filas_debug.append(fila_debug)

        #Dibujando...
        n_filas = (h_roi - CELL) // CELL + 1 if h_roi >= CELL else 0
        n_cols = (w_roi - CELL) // CELL + 1 if w_roi >= CELL else 0
        activado = np.zeros((n_filas, n_cols), dtype=np.uint8)

        for fila in range(n_filas):
            for col in range(n_cols):
                cy, cx = fila * CELL, col * CELL
                celda = fgDil[cy:cy + CELL, cx:cx + CELL]
                ocupacion = cv2.countNonZero(celda) / float(CELL * CELL)

                if ocupacion >= UMBRAL_ACTIVACION:
                    activado[fila, col] = 255
                    cv2.rectangle(roi, (cx, cy), (cx + CELL, cy + CELL), (0, 200, 255), 1)
                else:
                    cv2.rectangle(roi, (cx, cy), (cx + CELL, cy + CELL), (60, 60, 60), 1)

        #Contabilizando..
        celdas_activadas = int(np.count_nonzero(activado))
        densidad_banda = celdas_activadas / b["celdas_por_persona"]
        total_densidad += densidad_banda

        #Extra: ver las celdas activadas como componentes conectadas
        '''
        n_grupos, etiquetas, stats, _ = cv2.connectedComponentsWithStats(activado, connectivity=8)
        for etiqueta in range(1, n_grupos):  # 0 es el fondo (celdas apagadas)
            gx, gy, gw, gh, n_celdas_grupo = stats[etiqueta]
            px, py = gx * CELL, gy * CELL
            pw, ph = gw * CELL, gh * CELL

            personas_grupo = n_celdas_grupo / b["celdas_por_persona"]
            cv2.rectangle(roi, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
            cv2.putText(
                roi, f"{personas_grupo:.1f}", (px, max(0, py - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )
        '''

        cv2.rectangle(img, (b["x0"], b["y0"]), (b["x1"], b["y1"]), (0, 255, 255), 1)
        cv2.putText(
            img, f"{densidad_banda:.1f}",
            (b["x0"] + 5, b["y0"] + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2,
        )

    cv2.putText(
        img, f"Total rampa (densidad): {total_densidad:.1f}",
        (10, img.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )

    ancho_debug = max(f.shape[1] for f in filas_debug)
    filas_debug = [
        cv2.resize(f, (ancho_debug, f.shape[0])) if f.shape[1] != ancho_debug else f
        for f in filas_debug
    ]

    cv2.imshow("Original", original)
    cv2.imshow("video", img)
    cv2.imshow("Filtros por banda (fondo -> umbral -> mediana -> dilatar/abrir)",
               np.vstack(filas_debug))

    if cv2.waitKey(30) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
