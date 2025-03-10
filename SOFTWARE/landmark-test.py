import cv2
import mediapipe
from facial_landmarks import FaceLandmarks
import numpy as np
import os
import time
from open_cv_process import detect_and_track_faces
from background_management import get_random_background
from text_overlay import add_text_to_image

# Constantes
NO_FACE_DETECTED_MAX_ATTEMPTS = 3
BACKGROUND_REFRESH_TIME = 5  # secondes
BACKGROUND_SIZE = (1920, 1080)
WAITING_SCREEN_PATH = "background/waiting.jpg"

# Références des fonds avec position et ratio
BACKGROUND_REFS = {
    "dafoe.jpg": {"x_faceplacement": 0.2, "y_faceplacement": 0.5, "face_ratio": 0.15},
    "koons.jpg": {"x_faceplacement": 0.6, "y_faceplacement": 0.75, "face_ratio": 0.22},
    "predatwink.jpg": {"x_faceplacement": 0.45, "y_faceplacement": 0.5, "face_ratio": 0.15},
    "waiting.jpg": {"x_faceplacement": 0.8, "y_faceplacement": 0.1, "face_ratio": 0.6}
}

class FaceProcessingError(Exception):
    pass

def initialize_system():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise FaceProcessingError("Impossible d'ouvrir la caméra")
        
        fl = FaceLandmarks()
        waiting_image = cv2.imread(WAITING_SCREEN_PATH)
        if waiting_image is None:
            raise FaceProcessingError(f"Impossible de charger {WAITING_SCREEN_PATH}")
        
        selected_background = "dafoe.jpg"
        background_ref = BACKGROUND_REFS[selected_background]
        background = cv2.imread(f"background/{selected_background}")
        if background is None:
            raise FaceProcessingError(f"Impossible de charger {selected_background}")
        
        return cap, fl, waiting_image, background, background_ref
    except FaceProcessingError as e:
        print(f"Erreur d'initialisation: {e}")
        return None, None, None, None, None

def process_frame(frame, fl, background, background_ref, waiting_image, no_face_attempts):
    try:
        frame_copy = frame.copy()
        height, width, _ = frame.shape
        
        landmarks = fl.get_facial_landmarks(frame)
        if landmarks is None:
            no_face_attempts += 1
            return (waiting_image if no_face_attempts >= NO_FACE_DETECTED_MAX_ATTEMPTS else background), no_face_attempts
        
        no_face_attempts = 0
        convexhull = cv2.convexHull(landmarks)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, convexhull, 255)
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
        
        output = cv2.resize(background, BACKGROUND_SIZE)
        h_extended, w_extended = face_extracted.shape[:2]
        aspect_ratio = w_extended / h_extended
        
        target_height = int(output.shape[0] * background_ref['face_ratio'])
        target_width = int(target_height * aspect_ratio)
        
        face_masked = cv2.resize(face_extracted, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        face_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        bg_x = int(background_ref['x_faceplacement'] * output.shape[1] - target_width // 2)
        bg_y = int(background_ref['y_faceplacement'] * output.shape[0] - target_height // 2)
        
        bg_x = max(0, min(bg_x, output.shape[1] - target_width))
        bg_y = max(0, min(bg_y, output.shape[0] - target_height))
        
        region = output[bg_y:bg_y+target_height, bg_x:bg_x+target_width]
        result = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(face_mask))
        result = cv2.add(result, face_masked)
        output[bg_y:bg_y+target_height, bg_x:bg_x+target_width] = result
        
        return output, no_face_attempts
    except Exception as e:
        print(f"Erreur de traitement: {e}")
        return background, no_face_attempts

def main():
    cap, fl, waiting_image, background, background_ref = initialize_system()
    if any(v is None for v in [cap, fl, waiting_image, background, background_ref]):
        return
    
    no_face_attempts = 0
    last_background_refresh = time.time()
    
    try:
        while True:
            current_time = time.time()
            if current_time - last_background_refresh > BACKGROUND_REFRESH_TIME:
                selected_background = np.random.choice(list(BACKGROUND_REFS.keys()))
                background_ref = BACKGROUND_REFS[selected_background]
                background = cv2.imread(f"background/{selected_background}")
                if background is None:
                    print("Erreur: Impossible de charger le nouveau fond")
                    continue
                last_background_refresh = current_time
            
            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire le frame")
                break
            
            output, no_face_attempts = process_frame(frame, fl, background, background_ref, waiting_image, no_face_attempts)
            
            cv2.imshow("Resultat", output)
            
            if cv2.waitKey(30) == 27:
                break
    except KeyboardInterrupt:
        print("Arrêt par l'utilisateur")
    except Exception as e:
        print(f"Erreur critique: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
