import os
import cv2
import time
from open_cv_process import detect_and_track_faces
from background_management import get_random_background
from text_overlay import add_text_to_image  # Importer la fonction

background_refresh_time = 5

def main():
    """Fonction principale."""
    #  create neuronal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # background selection 
    selected_background, background_ref  = get_random_background("background")
    background = cv2.imread(f"background/{selected_background}")
    
    # timer init
    start_t = time.time()

    # video captureq
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        frame_copy = frame.copy()

        #background refreshing part
        if time.time() - start_t > background_refresh_time :
            selected_background, background_ref  = get_random_background("background")
            background = cv2.imread(f"background/{selected_background}")
            start_t = time.time()

        #cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #cv2.resizeWindow("result", 1920, 1080)
        
        frame, image = detect_and_track_faces(frame, face_cascade, background_ref, background)
        cv2.imshow("result", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()