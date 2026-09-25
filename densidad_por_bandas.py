import cv2
import numpy as np


video = cv2.VideoCapture("video_grafica.mp4")


Y0, Y1 = 365, 795
N_BANDAS = 3

# posición y ancho horizontal, independiente por banda: (x0, x1)
# debe tener exactamente N_BANDAS elementos, uno por banda (de
# arriba/lejos a abajo/cerca)
X_POR_BANDA = [
    (0, 380),
    (0, 380),
    (0, 380),
]

CAL_Y_A, CAL_CPP_A = 472, 5
CAL_Y_B, CAL_CPP_B = 687, 8


def celdas_por_persona_en(y_medio: float) -> int:
    """Interpola linealmente entre los dos puntos calibrados."""
    t = (y_medio - CAL_Y_A) / (CAL_Y_B - CAL_Y_A)
    cpp = CAL_CPP_A + t * (CAL_CPP_B - CAL_CPP_A)
    return max(1, round(cpp))



assert len(X_POR_BANDA) == N_BANDAS, "X_POR_BANDA debe tener un (x0,x1) por cada banda"

BANDAS = []
alto_banda = (Y1 - Y0) / N_BANDAS
for i in range(N_BANDAS):
    y0 = int(Y0 + i * alto_banda)
    y1 = int(Y0 + (i + 1) * alto_banda)
    y_medio = (y0 + y1) / 2
    x0_banda, x1_banda = X_POR_BANDA[i]
    BANDAS.append({
        "nombre": f"b{i}",
        "y0": y0, "y1": y1, "x0": x0_banda, "x1": x1_banda,
        "celdas_por_persona": celdas_por_persona_en(y_medio),
        "bg": cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True),
    })


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

    for b in BANDAS:
        # -----------------------------------------------------
        # 1) RECORTE DE LA BANDA
        # -----------------------------------------------------
        roi = img[b["y0"]:b["y1"], b["x0"]:b["x1"]]
        h_roi, w_roi = roi.shape[:2]

        # -----------------------------------------------------
        # 2) MOVIMIENTO (sustractor propio de la banda)
        # -----------------------------------------------------
        fg = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        #fg = b["bg"].apply(roi)
        #fg[fg == 127] = 0

        # -----------------------------------------------------
        # 3) LIMPIEZA
        # -----------------------------------------------------
        fgTH = cv2.adaptiveThreshold(
            fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -16
        )
        fgMedian = cv2.medianBlur(fgTH, 5)
        fgDil = cv2.dilate(fgMedian, kernel)
        fgDil = cv2.morphologyEx(fgDil, cv2.MORPH_OPEN, kernel_ruido)

        # Guardar las etapas para mostrarlas luego junto al video original
        fila_debug = np.hstack([
            etiquetar(fg, "1) fondo (MOG2)"),
            etiquetar(fgTH, "2) umbral adaptativo"),
            etiquetar(fgMedian, "3) mediana"),
            etiquetar(fgDil, "4) dilatar + abrir"),
        ])
        filas_debug.append(fila_debug)

        # -----------------------------------------------------
        # 4) CUADRÍCULA DE LA BANDA
        # -----------------------------------------------------
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

        celdas_activadas = int(np.count_nonzero(activado))
        densidad_banda = celdas_activadas / b["celdas_por_persona"]
        total_densidad += densidad_banda

        # -----------------------------------------------------
        # 4b) AGRUPAR CELDAS VECINAS ACTIVADAS -> "detecciones"
        #     Cada grupo de celdas conectadas (8-conectividad, para
        #     que fragmentos que solo se tocan en diagonal cuenten
        #     como el mismo grupo) se dibuja como UN rectángulo,
        #     para poder ver si ese grupo corresponde a una persona
        #     completa (~1.0) o a una detección incompleta/fragmento
        #     (bastante menos que 1.0).
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # 5) DIBUJAR LÍMITE DE LA BANDA Y SU ESTIMACIÓN
        # -----------------------------------------------------
        cv2.rectangle(img, (b["x0"], b["y0"]), (b["x1"], b["y1"]), (0, 255, 255), 1)
        cv2.putText(
            img, f"{densidad_banda:.1f}",
            (b["x0"] + 5, b["y0"] + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2,
        )

    # ---------------------------------------------------------
    # 6) ZONA IGNORADA (limitación documentada) + TOTAL
    # ---------------------------------------------------------
    cv2.putText(img, "(zona de cola no contada)", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(
        img, f"Total rampa (densidad): {total_densidad:.1f}",
        (10, img.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )

    cv2.imshow("Original", original)
    cv2.imshow("video", img)
    cv2.imshow("Filtros por banda (fondo -> umbral -> mediana -> dilatar/abrir)",
               np.vstack(filas_debug))

    if cv2.waitKey(30) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
