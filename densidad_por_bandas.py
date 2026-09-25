# =============================================================
# PROTOTIPO: DENSIDAD POR CUADRÍCULA, VARIAS BANDAS HORIZONTALES
# =============================================================
# Se descarta la banda "lejos" (cerca de la entrada, y=150-365):
# ahí la gente se detiene o camina muy lento formando una cola, y
# un sustractor de fondo (MOG2) termina aprendiendo a esas personas
# como si fueran parte del fondo -> deja de marcarlas como
# movimiento sin importar qué tan fina sea la cuadrícula. Esto no
# es un problema de resolución de grilla, es que ya no hay señal
# de "movimiento" para medir ahí. Se documenta como limitación y
# no se cuenta esa zona.
#
# El resto de la rampa (y=365 a y=795, donde la gente sí camina de
# forma continua) se divide en tantas bandas horizontales como
# N_BANDAS indique, para compensar la perspectiva: una persona
# cerca de la cámara ocupa más celdas que una persona lejos.
#
# CELDAS_POR_PERSONA por banda se interpola linealmente entre dos
# puntos calibrados a ojo viendo la cuadrícula sobre el video real:
#   y=472 (centro de la vieja banda "medio") -> ~5 celdas/persona
#   y=687 (centro de la vieja banda "cerca") -> ~8 celdas/persona
# No es un modelo físico de la cámara, es una aproximación simple
# a partir de dos mediciones reales.
#
# Sigue usando solo procesamiento de imagen (sustracción de fondo +
# umbralización + morfología): nada de aprendizaje automático.
# =============================================================
import cv2
import numpy as np

# =============================================================
# CAPTURA DE VIDEO
# =============================================================
video = cv2.VideoCapture("video_grafica.mp4")

# =============================================================
# RANGO DE LA RAMPA A CONTAR (se excluye la zona de cola/entrada)
# =============================================================
Y0, Y1 = 365, 795
X0, X1 = 0, 380          # excluye la baranda/escalera de la derecha
N_BANDAS = 3              # cada banda debe ser más alta que una persona
                           # (ver calibración: alturas típicas de ~30-60px,
                           # con picos bastante más grandes por fragmentos
                           # que se funden entre sí) para no cortarla a la
                           # mitad entre dos bandas.

# --- Puntos de calibración (ver comentario arriba) ---
CAL_Y_A, CAL_CPP_A = 472, 5
CAL_Y_B, CAL_CPP_B = 687, 8


def celdas_por_persona_en(y_medio: float) -> int:
    """Interpola linealmente entre los dos puntos calibrados."""
    t = (y_medio - CAL_Y_A) / (CAL_Y_B - CAL_Y_A)
    cpp = CAL_CPP_A + t * (CAL_CPP_B - CAL_CPP_A)
    return max(1, round(cpp))


# =============================================================
# CONSTRUIR LAS BANDAS
# =============================================================
BANDAS = []
alto_banda = (Y1 - Y0) / N_BANDAS
for i in range(N_BANDAS):
    y0 = int(Y0 + i * alto_banda)
    y1 = int(Y0 + (i + 1) * alto_banda)
    y_medio = (y0 + y1) / 2
    BANDAS.append({
        "nombre": f"b{i}",
        "y0": y0, "y1": y1, "x0": X0, "x1": X1,
        "celdas_por_persona": celdas_por_persona_en(y_medio),
        "bg": cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True),
    })

# =============================================================
# KERNELS
# =============================================================
kernel = np.ones((5, 5), np.uint8)
kernel_ruido = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# =============================================================
# PARÁMETROS DE LA CUADRÍCULA
# =============================================================
CELL = 20
UMBRAL_ACTIVACION = 0.35

# =============================================================
# ESTABILIZACIÓN (opcional del enunciado)
# =============================================================
# Se midió el corrimiento real entre frames con cv2.phaseCorrelate
# sobre ~1000 frames: mediana ~0.6px, promedio ~1.0px. Es un temblor
# pequeño de una cámara fija (vibración de la estructura / compresión
# del video), no un paneo.
#
# Se compara cada frame contra el frame ANTERIOR (no contra uno fijo
# del inicio del video): en un video de 5-10 minutos la multitud es
# completamente distinta minuto a minuto, así que comparar contra un
# solo frame de referencia lejano vuelve la correlación de fase poco
# fiable (se probó y el "error" resultante era enorme, ~200px, solo
# por el contenido tan distinto, no por temblor real). Frame a frame
# el contenido cambia poco, así que la comparación es confiable.
#
# cv2.phaseCorrelate(anterior, actual) devuelve el corrimiento (dx,dy)
# tal que actual ≈ anterior desplazado por (dx,dy); para alinear
# actual de vuelta hay que aplicarle el desplazamiento INVERSO
# (-dx,-dy) (se verificó con una prueba controlada: un corrimiento
# conocido de (10,5) devuelve (dx,dy)=(10,5), no su negativo).
#
# cv2.phaseCorrelate mide correlación global de todo el frame, así
# que una multitud grande moviéndose puede "engañarlo" y devolver un
# corrimiento gigante (se vio un caso de 50px) que NO es la cámara
# moviéndose. Por eso se descarta cualquier corrimiento mayor a
# MAX_DESPLAZAMIENTO: un temblor real de cámara es de pocos píxeles,
# así que un valor grande es señal de que la estimación es ruido.
# =============================================================
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
        fg = b["bg"].apply(roi)
        fg[fg == 127] = 0

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
