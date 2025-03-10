import cv2
import numpy as np

background_size = (1920, 1080)

# Configuration de la zone à exclure (ajouté)
EXCLUSION_ZONE = {
    'x': 100,    # Position X de la zone à exclure
    'y': 100,    # Position Y de la zone à exclure
    'w': 200,    # Largeur de la zone à exclure
    'h': 200     # Hauteur de la zone à exclure
}

def detect_and_track_faces(frame, face_cascade, img_coordonate, background):
    """Détection et suivi des visages."""
    #output waiting background
    waiter = cv2.resize(cv2.imread(f"background/waiting.jpg"), background_size)
    output = waiter
    frame_copy = frame.copy()
    frame_copy = cv2.blur(frame_copy, (27, 27))

    # Conversion en gris
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    
    mask = np.ones(gray.shape, dtype=np.uint8) * 255

    # Création d'un masque pour exclure la zone 
    # DISCARD FOR MANONO LAPTOP 
    # cv2.rectangle(mask, 
    #              (EXCLUSION_ZONE['x'], EXCLUSION_ZONE['y']), 
    #              (EXCLUSION_ZONE['x'] + EXCLUSION_ZONE['w'], 
    #               EXCLUSION_ZONE['y'] + EXCLUSION_ZONE['h']), 
    #              0, -1)
    
    # Appliquer le masque à l'image en gris
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Détecter les visages sur l'image masquée
    faces = face_cascade.detectMultiScale(gray_masked, 1.3, 5)
    
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
        face_elliptical = cv2.resize(face_elliptical, (resized_dia, resized_dia), 
                           interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask,(resized_dia,resized_dia))
        contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, (0,255,0), 1)

        # Placement sur l'arrière-plan
        bg_x = int(img_coordonate['x_faceplacement']*output.shape[1]-face_elliptical.shape[1]//2)
        bg_y = int(img_coordonate['y_faceplacement']*output.shape[0]-face_elliptical.shape[0]//2)

        # face addition to background (prone to error if face is oversized !!)
        region = output[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])]
        result = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(mask))
        result = cv2.add(result, face_elliptical)
        output[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])] = result
        
        
        cv2.imshow("elicitp", mask)
    return frame, output