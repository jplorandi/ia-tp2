import cv2
import numpy as np

def hough_lines(image, gray):
    #Uso Canny para detectar bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    #Detecto Hough via OpenCV
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    # Ploteo las lineas encontradas
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Muestro resultado x pantalla
    cv2.imshow(f'Detectadas {len(lines)} lineas.', image)
    loop = True
    while loop:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def hough_circles(image, gray):
    # Aplicamos un blur para reducir ruido
    gray_blurred = cv2.medianBlur(gray, 5)

    # Detectar circulos con la transformada de Hough de Circulos
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=1,
        maxRadius=150
    )

    # Ploteamos todos los circulos detectados
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])  # Circle center
            radius = circle[2]  # Circle radius
            # Draw the circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # Draw the circle outline
            cv2.circle(image, center, radius, (255, 0, 255), 3)

    # Mostramos resultados
    cv2.imshow(f'Detectados {len(circles)} Circulos', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread("./images/hough/test-image.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hough_lines(image.copy(), gray.copy())
    hough_circles(image.copy(), gray.copy())
