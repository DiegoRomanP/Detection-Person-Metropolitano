import cv2

cap = cv2.VideoCapture("video_grafica.mp4")
ret, frame = cap.read()  # primer frame
cap.release()

# Seleccionar ROI interactivamente
roi = cv2.selectROI("Selecciona el area", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = roi
print(f"x={x}, y={y}, ancho={w}, alto={h}")
print(f"Esquina superior izquierda: ({x}, {y})")
print(f"Esquina inferior derecha:   ({x + w}, {y + h})")
