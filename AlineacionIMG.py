# Importamos librerias
import numpy as np
import cv2

# Creamos la Video Captura
cap = cv2.VideoCapture(0)

# Leer la imagen original
im1 = cv2.imread('frontal.jpg')
ancho = int(im1.shape[1]/5)
alto = int(im1.shape[0]/5)
im1 = cv2.resize(im1, (ancho, alto), interpolation = cv2.INTER_AREA)

# Creamos un ciclo para ejecutar nuestros Frames
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()

    # Convertimos a EDG
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Buscamos puntos claves
    # Numero de puntos clave
    num_kpt = 500
    # Declaramos el objeto
    orb = cv2.ORB_create(num_kpt)
    # Extraemos la info de la img
    keypoint1, descriptor1 = orb.detectAndCompute(gray_im1, None)
    # Extraemos la info de los frames
    keypoint2, descriptor2 = orb.detectAndCompute(gray_frame, None)


    # Dibujamos
    im1_display = cv2.drawKeypoints(im1, keypoint1, outImage = np.array([]), color =(255,0,0),
                                    flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # ¿Como hacemos coincidir los puntos?
    # 1. Creamos un objeto comparador de descriptores
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(descriptor1, descriptor2)

    # 2. Ordenamos la lista
    matches = sorted(matches, key = lambda x: x.distance, reverse = False)

    # 3. Filtramos los resultados
    goodmatches = int(len(matches) * 0.1)
    matches = matches[:goodmatches]

    # 4. Mostramos las coincidencias
    img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

    # ¿Como calculamos la homografia de la imagen?
    # 1. Creamos listas con el tamaño del total de keypoints
    puntos1 = np.zeros((len(matches), 2), dtype = np.float32)
    puntos2 = np.zeros((len(matches), 2), dtype = np.float32)

    # 2. Extraemos los puntos
    for i, match in enumerate(matches):
        # Puntos de imagen
        puntos1[i, :] = keypoint1[match.queryIdx].pt
        # Puntos de frames
        puntos2[i, :] = keypoint2[match.trainIdx].pt

    # 3. Extraemos la homografia
    h, mask = cv2.findHomography(puntos2, puntos1, cv2.RANSAC)

    # 4. Dibujamos
    alto, ancho, canales = im1.shape
    img_perspec = cv2.warpPerspective(frame, h, (ancho, alto))



    # Mostramos caracteristicas
    cv2.imshow("VIDEO CAPTURA", frame_display)
    cv2.imshow("IMAGEN", im1_display)
    cv2.imshow("COINCIDENCIAS", img_perspec)

    # Cerramos con lectura de teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()