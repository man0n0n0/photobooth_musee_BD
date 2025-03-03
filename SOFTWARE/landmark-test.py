import cv2
import mediapipe
from facial_landmarks import FaceLandmarks
import numpy as np
import os
import time
from open_cv_process import detect_and_track_faces
from background_management import get_random_background
from text_overlay import add_text_to_image  # Importer la fonction

fl = FaceLandmarks()

cap = cv2.VideoCapture(0)

background_refresh_time = 5
background_size = (1920, 1080)

# Sélectionner un fond aléatoire au démarrage
selected_background, background_ref = get_random_background("background")
background = cv2.imread(f"background/{selected_background}")

while True:
    # timer init
    start_t = time.time()
    
    # Changer de fond périodiquement
    if time.time() - start_t > background_refresh_time:
        selected_background, background_ref = get_random_background("background")
        background = cv2.imread(f"background/{selected_background}")
        start_t = time.time()
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.9, fy=0.9)
    frame_copy = frame.copy()
    height, width, _ = frame.shape
    landmarks = fl.get_facial_landmarks(frame)

    convexhull = cv2.convexHull(landmarks)

    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, convexhull, 255)

    #extract face
    frame_copy = cv2.blur(frame_copy, (27, 27))
    face_extracted = cv2.bitwise_and(frame_copy,frame_copy, mask=mask)

    blurred_face = cv2.GaussianBlur(face_extracted, (0, 0), sigmaX=5.0)

    # Créer une image finale avec le fond choisi
    output = cv2.resize(background, background_size)
    
    # Calculer les coordonnées de placement
    bg_x = int(background_ref['x_faceplacement']*output.shape[1]-blurred_face.shape[1]//2)
    bg_y = int(background_ref['y_faceplacement']*output.shape[0]-blurred_face.shape[0]//2)

    # Redimensionner le visage selon le ratio spécifié
    nouvelle_largeur = int(output.shape[1] * background_ref["face_ratio"])
    nouvelle_hauteur = int((blurred_face.shape[0] / blurred_face.shape[1]) * nouvelle_largeur)
    blurred_face = cv2.resize(blurred_face, (nouvelle_largeur, nouvelle_hauteur))

    # Placer le visage flou sur le fond
    region = output[bg_y:bg_y+int(blurred_face.shape[0]), bg_x:bg_x+int(blurred_face.shape[1])]
    
    # Assurer que le masque a la bonne taille et le bon type
    mask = cv2.resize(mask, (blurred_face.shape[1], blurred_face.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8)
    
    # Appliquer les opérations bit à bit
    result = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(mask))
    result = cv2.add(result, blurred_face)
    output[bg_y:bg_y+int(blurred_face.shape[0]), bg_x:bg_x+int(blurred_face.shape[1])] = result

    cv2.imshow("Resultat", output)

    key = cv2.waitKey(30)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()