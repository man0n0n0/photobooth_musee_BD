import cv2
import numpy as np

def detect_and_track_faces(frame, face_cascade, img_coordonate, background):
    """Détection et suivi des visages."""
    #backgorund operation
    current_background = cv2.resize(background, (1080, 1920))

    # Conversion en gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        center = x + w//2, y + h//2
        
        # Création du masque elliptique au lieu du cercle
        mask = np.zeros((max(w,h),max(w,h)), dtype=np.uint8)
        # Utilisation de cv2.ellipse avec l'axe majeur=w et mineur=h/2
        cv2.ellipse(
            mask,
            (max(w,h)//2, max(w,h)//2),
            (w//3, h//2),  # Axe majeur=w/2, axe mineur=h/4
            0,  # Pas d'angle de rotation
            0,  # Angle de début
            360,  # Angle de fin
            255,  # Couleur blanche pour le masque
            -1   # Épaisseur -1 pour remplir l'ellipse
        )
        face_elliptical = cv2.bitwise_and(face, face, mask=mask)
        
        # Redimensionnement
        resized_dia = int(current_background.shape[0]*img_coordonate['face_ratio'])
        face_elliptical = cv2.resize(face_elliptical,(resized_dia,resized_dia))
        mask = cv2.resize(mask,(resized_dia,resized_dia))

        # Placement sur l'arrière-plan
        bg_x = int(img_coordonate['x_faceplacement']*current_background.shape[1]-face_elliptical.shape[1])
        bg_y = int(img_coordonate['y_faceplacement']*current_background.shape[0]-face_elliptical.shape[0])

        if bg_y+face_elliptical.shape[0] < current_background.shape[0] and bg_x+face_elliptical.shape[1] < current_background.shape[1]:
            region = current_background[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])]
            result = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(mask))
            result = cv2.add(result, face_elliptical)
            current_background[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])] = result
        
        cv2.imshow("mask", result)

    
    return frame, current_background