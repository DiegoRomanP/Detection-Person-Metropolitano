# =============================================================
# LA TÉCNICA DEL PROFESOR, CON Y SIN MOG2, SOBRE CAJAS A MANO
# =============================================================
# Mismo pipeline base que professor_scripts/main.py:
#     -> adaptiveThreshold -> medianBlur -> dilate -> countNonZero
# aplicado sobre 3 cajas fijas en vez de espacios de parqueo, en dos
# versiones que corren en paralelo sobre el mismo frame:
#
#   SIN MOG2: exactamente el pipeline del profesor (gris -> umbral
#             adaptativo), sin aislar movimiento.
#   CON MOG2: primero se aísla el movimiento (bg.apply) y RECIÉN
#             ahí se aplica el mismo umbral adaptativo + limpieza,
#             igual que en contador_personas.py.
#
# La idea es comparar, en las MISMAS 3 cajas, qué tanto ayuda (o no)
# agregar la sustracción de fondo antes del umbral del profesor.
#
# Las cajas se sacaron de C:\Users\USER\Downloads\areaspng.png (3
# rectángulos rojos dibujados a mano sobre una captura del video),
# reescaladas de esa captura (823x846px) al video real (478x850px),
# y luego corregidas a mano por el usuario viendo el resultado.
#
# La caja de "arriba" cae dentro de la zona de cola/entrada
# (y=150-365) que en densidad_por_bandas.py se descartó porque el
# MOG2 no detecta gente parada/lenta ahí. Sirve para comparar ahí
# mismo si el profesor sin MOG2 se comporta mejor o peor.
# =============================================================
import cv2
import numpy as np

video = cv2.VideoCapture("video_grafica.mp4")
bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=40, detectShadows=True)

# (x, y, w, h) en coordenadas del video real (478x850)
CAJAS = [
    ("arriba (zona de cola)", 133, 214, 122, 125),
    ("medio", 122, 411, 145, 160),
    ("abajo", 85, 589, 223, 190),
]

kernel = np.ones((5, 5), np.uint8)

# =============================================================
# DENSIDAD: convertir píxeles activados en una estimación de personas
# =============================================================
# Calibrado viendo el mismo frame de muestra (frame ~700), contando
# personas reales a ojo en "arriba" y "medio" (~4 en cada una) y
# dividiendo por el count de cada pipeline ahí. Es una calibración de
# UN solo frame para cada pipeline (no una medición robusta), pero
# alcanza para comparar las dos técnicas entre sí:
#
#   SIN MOG2: arriba=8199/4≈2050   medio=6575/4≈1644 px^2/persona
#   CON MOG2: arriba=1648/4≈412    medio=4034/4≈1008 px^2/persona
#
# Dato interesante: con MOG2 el valor sube de "arriba" (lejos) a
# "medio" (cerca), como se espera por perspectiva (más cerca = más
# grande). Sin MOG2 pasa lo contrario -> el umbral del profesor no
# está midiendo tamaño de persona de forma confiable, sino textura en
# general, y esa textura no varía con la distancia de forma
# consistente. Otro argumento a favor de agregar MOG2.
PIXELES_POR_PERSONA_SIN_MOG2 = {"arriba (zona de cola)": 2050, "medio": 1650, "abajo": 1850}
PIXELES_POR_PERSONA_CON_MOG2 = {"arriba (zona de cola)": 400, "medio": 1000, "abajo": 700}

while True:
    check, img = video.read()
    if not check:
        break

    # ===========================================================
    # SIN MOG2: técnica del profesor, sin cambios
    # ===========================================================
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(
        imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    imgMedian = cv2.medianBlur(imgTH, 5)
    imgDil = cv2.dilate(imgMedian, kernel)

    # ===========================================================
    # CON MOG2: aislar movimiento primero, y RECIÉN AHÍ aplicar el
    # mismo tipo de umbral + limpieza (igual que contador_personas.py)
    # ===========================================================
    fg = bg.apply(img)
    fg[fg == 127] = 0  # descartar sombras
    fgTH = cv2.adaptiveThreshold(
        fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -16
    )
    fgMedian = cv2.medianBlur(fgTH, 5)
    fgDil = cv2.dilate(fgMedian, kernel)

    # ===========================================================
    # CONTAR PÍXELES ACTIVADOS EN CADA CAJA, EN AMBAS VERSIONES
    # ===========================================================
    video_cajas = img.copy()
    for nombre, x, y, w, h in CAJAS:
        count_sin = cv2.countNonZero(imgDil[y:y + h, x:x + w])
        count_con = cv2.countNonZero(fgDil[y:y + h, x:x + w])

        personas_sin = count_sin / PIXELES_POR_PERSONA_SIN_MOG2[nombre]
        personas_con = count_con / PIXELES_POR_PERSONA_CON_MOG2[nombre]

        cv2.rectangle(video_cajas, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(video_cajas, f"{nombre}",
                    (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(video_cajas, f"sin MOG2: {personas_sin:.1f}",
                    (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(video_cajas, f"con MOG2: {personas_con:.1f}",
                    (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imshow("video (cajas: sin MOG2 vs con MOG2)", video_cajas)
    cv2.imshow("mascara SIN MOG2 (umbral del profesor)", imgDil)
    cv2.imshow("mascara CON MOG2", fgDil)

    if cv2.waitKey(30) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
