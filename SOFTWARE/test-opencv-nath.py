import cv2
import numpy as np
import random
import os

def get_random_background(background_folder):
    """Sélectionne une image aléatoire du dossier."""
    if not os.path.exists(background_folder):
        raise FileNotFoundError(f"Dossier '{background_folder}' n'existe pas !")
    
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        f for f in os.listdir(background_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
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
        radius = max(w,h)//2
        #cv2.circle(frame, center, radius, (0,0,0), 1)
        
        # Redimensionnement et masquage circulaire
        face_resized = cv2.resize(face, (150, 150))
        mask = np.zeros((150, 150), dtype=np.uint8)
        cv2.circle(mask, (95, 75), 75, 255, -1)
        face_circular = cv2.bitwise_and(face_resized, face_resized, mask=mask)
        
        # Placement sur le fond
        bg_x = img_coordonate['img1x']
        bg_y = img_coordonate['img1y']
        
        if bg_y + 150 < current_background.shape[0] and bg_x + 150 < current_background.shape[1]:
            region = current_background[bg_y:bg_y+150, bg_x:bg_x+150]
            result = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(mask))
            result = cv2.add(result, face_circular)
            current_background[bg_y:bg_y+150, bg_x:bg_x+150] = result
            
    return frame, current_background

def main():
    """Fonction principale."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Chargement et préparation du fond
    background = cv2.imread(get_random_background("background"))
    current_background = cv2.resize(background.copy(), (1080, 1920))
    
    # Configuration des coordonnées
    img_coordonate = {'img1x': 550, 'img1y': 550}
    
    # Capture vidéo
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        frame, current_background = detect_and_track_faces(frame, face_cascade, img_coordonate, current_background)
        cv2.imshow("image", current_background)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
