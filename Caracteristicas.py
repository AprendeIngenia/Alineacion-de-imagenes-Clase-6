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

    #print(descriptor1)


    # Dibujamos
    im1_display = cv2.drawKeypoints(im1, keypoint1, outImage = np.array([]), color =(255,0,0),
                                    flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)


    # Mostramos caracteristicas
    cv2.imshow("VIDEO CAPTURA", frame_display)
    cv2.imshow("IMAGEN", im1_display)

    # Cerramos con lectura de teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()