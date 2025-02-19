import cv2
import numpy as np

def get_random_background(background_folder):
    """Get a random image from the specified folder."""
    if not os.path.exists(background_folder):
        raise FileNotFoundError(f"Background folder '{background_folder}' does not exist!")
        
    # Get list of all image files in the folder
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        f for f in os.listdir(background_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No valid image files found in '{background_folder}'!")
        
    # Select random image -- modify 
    selected_image = random.choice(image_files)
    return selected_image



def main():

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    #background = cv2.imread("dafoe.jpg")
    background = get_random_background("background")

    copy_background = background.copy()
    current_background = background.copy()


    scale_up = 0.5
     
    # current_background = cv2.resize(copy_background, None, fx= scale_up, fy= scale_up, interpolation= cv2.INTER_LINEAR)

    img_coordonate = {}

    # Detect bg visage -- to delete
    pin_faces = face_cascade.detectMultiScale(current_background, 1.3, 5)

    img_coordinates = {}
    for (x, y, w, h) in pin_faces:
        img_coordonate['img1x'] = x
        img_coordonate['img1y'] = y
        
        print(img_coordonate['img1x'],img_coordonate['img1y'] )


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            #cv2.rectangle(frame,(x,y), (x+w,y+h), (0,0,0), 1)
            center_coordinates = x + w//2, y + h//2
            radius = max(w,h)//2
            cv2.circle(frame, center_coordinates, radius, (0,0,0), 1)
            
            face_resized = cv2.resize(face, (150, 150))
            
            # placement sur le fond 
            bg_x = img_coordonate['img1x']
            bg_y = img_coordonate['img1y']
            
            if bg_y + 150 < current_background.shape[0] and bg_x + 150 < current_background.shape[1]:
                current_background[bg_y:bg_y+150, bg_x:bg_x+150] = face_resized

        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("image", 1270, 720)

        cv2.imshow("image", current_background)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()