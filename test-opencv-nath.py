import cv2
import numpy as np
import random
import os

def get_random_background(background_folder):
    """Sélectionne une image aleatoire du dossier."""
    if not os.path.exists(background_folder):
        raise FileNotFoundError(f"Dossier '{background_folder}' n'existe pas !")
    
    #exclude specific wainting image files and invalid extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    waiting_images = ['waiting.jpg','waiting.jpg']
    image_files = [
        f for f in os.listdir(background_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions 
        and f not in waiting_images
    ]

    if not image_files:
        raise ValueError(f"Aucun fichier image trouvé dans '{background_folder}' !")
    
    return random.choice(image_files)

def detect_and_track_faces(frame, face_cascade, img_coordonate, current_background):
    """Détection et suivi des visages."""
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

def main():
    """Fonction principale."""
    # definir les references de placement des visages pour chaque image d'arrière plan (dans le dossier background)
    background_refs = {
        "dafoe.jpg" : {
        "x_faceplacement" : 0.5,
        "y_faceplacement" : 0.5,
        "face_ratio" : 0.1
        },
        "koons.jpg" : {
        "x_faceplacement" : 0.5,
        "y_faceplacement" : 0.5,
        "face_ratio" : 0.2
        },
        "predatwink.jpg" : {
        "x_faceplacement" : 0.5,
        "y_faceplacement" : 0.5,
        "face_ratio" : 0.05
        },
    } 

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Chargement et préparation du fond
    selected_background = get_random_background("background")
    background = cv2.imread(f"background/{selected_background}")
    current_background = cv2.resize(background, (1080, 1920))
    
    # video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        frame, current_background = detect_and_track_faces(frame, face_cascade, background_refs[selected_background], current_background)
        cv2.imshow("image", current_background)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
