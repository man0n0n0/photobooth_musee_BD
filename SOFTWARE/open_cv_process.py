import cv2
import numpy as np

background_size = (1080, 1920)

def detect_and_track_faces(frame, face_cascade, img_coordonate, background):
    
    """Détection et suivi des visages."""
    #output waiting background
    waiter = cv2.resize(cv2.imread(f"background/waiting.jpg"),background_size)
    output = waiter

    # Conversion en gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        output = cv2.resize(background, background_size)

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
        resized_dia = int(output.shape[0]*img_coordonate['face_ratio'])
        face_elliptical = cv2.resize(face_elliptical,(resized_dia,resized_dia))
        mask = cv2.resize(mask,(resized_dia,resized_dia))

        # Placement sur l'arrière-plan
        bg_x = int(img_coordonate['x_faceplacement']*output.shape[1]-face_elliptical.shape[1]//2)
        bg_y = int(img_coordonate['y_faceplacement']*output.shape[0]-face_elliptical.shape[0]//2)

        # face addition to background (prone to error if face is oversized !!)
        region = output[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])]
        result = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(mask))
        result = cv2.add(result, face_elliptical)
        output[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])] = result


    return frame, output