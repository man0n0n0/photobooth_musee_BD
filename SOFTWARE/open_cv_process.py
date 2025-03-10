import cv2
import dlib
import numpy as np
import random

background_size = (1920, 1080)

# Configuration de la zone à exclure
EXCLUSION_ZONE = {
    'x': 100,    # Position X de la zone à exclure
    'y': 100,    # Position Y de la zone à exclure
    'w': 200,    # Largeur de la zone à exclure
    'h': 200     # Hauteur de la zone à exclure
}

# Initialize these only once outside the functions
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def create_landmarks_mask(face, rect):
    """
    Create a mask for the face using facial landmarks from dlib.
    
    Args:
        face (numpy.ndarray): The face image
        rect (dlib.rectangle): The face rectangle from dlib
    
    Returns:
        numpy.ndarray: A binary mask following the face contour
    """
    # Create an empty mask with the same size as the face
    mask = np.zeros(face.shape[:2], dtype=np.uint8)
    
    # If no face rectangle provided, return a default elliptical mask
    if rect is None:
        h, w = face.shape[:2]
        cv2.ellipse(
            mask,
            (w//2, h//2),
            (w//2, h//2),
            0, 0, 360, 255, -1
        )
        return mask
    
    # Get facial landmarks
    shape = shape_predictor(face, rect)
    points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int32)
    
    # Get the face contour points
    face_contour = np.vstack([
        points[0:17],  # Jawline
        points[26:16:-1],  # Right eyebrow (reversed)
        points[0:1]  # Connect back to the start
    ])
    
    # Draw the face contour as a filled polygon
    cv2.fillPoly(mask, [face_contour], 255)
    
    return mask

def detect_and_track_faces(frame, face_cascade, img_coordonate, background):
    """Détection et suivi des visages avec masque basé sur les points de repère faciaux."""
    # Pre-load and resize background images
    waiter = cv2.resize(cv2.imread(f"background/waiting.jpg"), background_size)
    resized_background = cv2.resize(background, background_size)
    output = waiter
    
    # Downscale frame for faster processing
    scale_factor = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Convert to grayscale once
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Create exclusion mask
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    
    # Apply the mask
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Use faster detection parameters
    faces = face_cascade.detectMultiScale(gray_masked, 1.2, 3, minSize=(30, 30))
    
    # If no faces detected with Haar, return waiting image
    if len(faces) == 0:
        return frame, output
    
    # Select only largest face for processing
    if len(faces) > 1:
        face_sizes = [w*h for (x,y,w,h) in faces]
        largest_face_idx = face_sizes.index(max(face_sizes))
        faces = [faces[largest_face_idx]]
    
    # Process the selected face
    for (x, y, w, h) in faces:
        # Scale coordinates back to original frame size
        x, y, w, h = int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)
        
        # Extract face region
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue
            
        # Convert face to grayscale for dlib
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect face with dlib
        rect = None
        dlib_faces = face_detector(gray_face, 0)  # Use 0 for faster detection
        if len(dlib_faces) > 0:
            rect = dlib_faces[0]
        
        # Create mask
        face_mask = create_landmarks_mask(gray_face, rect)
        
        # Apply mask to face
        face_masked = cv2.bitwise_and(face, face, mask=face_mask)
        
        # Calculate output dimensions once
        resized_dia = int(resized_background.shape[0] * img_coordonate['face_ratio'])
        
        # Resize face and mask
        face_masked = cv2.resize(face_masked, (resized_dia, resized_dia), interpolation=cv2.INTER_LINEAR)
        face_mask = cv2.resize(face_mask, (resized_dia, resized_dia), interpolation=cv2.INTER_LINEAR)
        
        # Calculate placement once
        bg_x = int(img_coordonate['x_faceplacement'] * resized_background.shape[1] - resized_dia//2)
        bg_y = int(img_coordonate['y_faceplacement'] * resized_background.shape[0] - resized_dia//2)
        
        # Ensure coordinates are within bounds
        if (bg_x < 0 or bg_y < 0 or 
            bg_x + resized_dia > resized_background.shape[1] or 
            bg_y + resized_dia > resized_background.shape[0]):
            continue
        
        # Extract region
        region = resized_background[bg_y:bg_y+resized_dia, bg_x:bg_x+resized_dia]
        
        # Apply mask and add face
        inv_mask = cv2.bitwise_not(face_mask)
        result = cv2.bitwise_and(region, region, mask=inv_mask)
        result = cv2.add(result, face_masked)
        
        # Update output
        output = resized_background.copy()
        output[bg_y:bg_y+resized_dia, bg_x:bg_x+resized_dia] = result
        
    return frame, output