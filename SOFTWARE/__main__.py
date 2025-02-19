import cv2
import os
import random

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
        
    # Select random image
    selected_image = random.choice(image_files)
    return cv2.imread(os.path.join(background_folder, selected_image))

def overlay_face_on_background(face_frame, background):
    """Overlay the face detection frame onto the background."""
    # Ensure background is large enough
    max_size = max(face_frame.shape[:2])
    if background.shape[0] < max_size or background.shape[1] < max_size:
        # Resize background if too small
        scale = max(max_size/background.shape[0], max_size/background.shape[1])
        background = cv2.resize(background, None, fx=scale, fy=scale)
    
    # Calculate position to center the face detection frame
    y_offset = (background.shape[0] - face_frame.shape[0]) // 2
    x_offset = (background.shape[1] - face_frame.shape[1]) // 2
    
    # Create ROI and overlay
    roi = background[y_offset:y_offset+face_frame.shape[0],
                    x_offset:x_offset+face_frame.shape[1]]
    result = cv2.addWeighted(roi, 0.5, face_frame, 0.5, 0)
    background[y_offset:y_offset+face_frame.shape[0],
              x_offset:x_offset+face_frame.shape[1]] = result
    
    return background

# Main program
def main():
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Specify background folder path
    background_folder = './background_images'  # Change this to your folder path
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            # Get random background
            background = get_random_background(background_folder)
            
            # Overlay face detection onto background
            result = overlay_face_on_background(frame, background.copy())
            
            # Display the resulting frame
            cv2.imshow('Face Detection with Background', result)
            
            # Press 'ESC' to exit
            if cv2.waitKey(30) & 0xff == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()