import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

background = cv2.imread("dafoe.jpg")
copy_background = background.copy()
current_background = background.copy()


scale_up = 0.5
 
# current_background = cv2.resize(copy_background, None, fx= scale_up, fy= scale_up, interpolation= cv2.INTER_LINEAR)

img_coordonate = {}

# Detect bg visage
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

    cv2.namedWindow("back", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("back", 1270, 720)


    cv2.imshow("webcam", frame)
    cv2.imshow("back", current_background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()