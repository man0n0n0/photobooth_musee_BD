import cv2
import dlib
import numpy as np
import time
from functools import lru_cache

# Global constants
BACKGROUND_SIZE = (720, 1040)

# Initialize face detection tools once
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Pre-compute common kernels
SMOOTH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Cache for background images
background_cache = {}

@lru_cache(maxsize=32)
def get_cached_background(bg_path):
    """Cache background images to avoid repeated disk I/O and resize operations"""
    if bg_path not in background_cache:
        bg_img = cv2.imread(bg_path)
        if bg_img is not None:
            background_cache[bg_path] = cv2.resize(bg_img, BACKGROUND_SIZE)
    return background_cache.get(bg_path)

def create_landmarks_mask(face, rect):
    """
    Create a mask for the face using facial landmarks with a rounded forehead.
    
    Args:
        face (numpy.ndarray): The face image
        rect (dlib.rectangle): The face rectangle from dlib
    
    Returns:
        numpy.ndarray: A binary mask following the face contour
    """
    # Create an empty mask
    h, w = face.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # If no face rectangle provided, return an empty mask
    if rect is None:
        return mask
    
    # Get facial landmarks
    shape = shape_predictor(face, rect)
    points = np.zeros((shape.num_parts, 2), dtype=np.int32)
    for i in range(shape.num_parts):
        points[i] = [shape.part(i).x, shape.part(i).y]
    
    # Get jawline points
    jawline = points[0:17]

    # Simple approach: calculate the center top point of the forehead
    left_temple = points[0]
    right_temple = points[16]

    # Calculate the width of the face at the temples
    face_width = right_temple[0] - left_temple[0]

    # Calculate the center point between temples
    face_center = ((left_temple[0]+right_temple[0])//2,(left_temple[1]+right_temple[1])//2)

    # Calculate forehead height (as proportion of face width)
    forehead_height = int(face_width * 0.6)

    # Calculate the angle 
    delta_temple_y = right_temple[1] - left_temple[1]
    face_angle = 90 - np.rad2deg(np.arctan(face_width/(delta_temple_y))) if delta_temple_y < 0 else 270 - abs(np.rad2deg(np.arctan(face_width/(delta_temple_y))))

    # Create a simple arc for the forehead-half-ellipse spanning from left temple to right temple
    cv2.ellipse(
        mask,
        face_center,  # center point (at eyebrow level)
        (face_width // 2, forehead_height),  # half width and height of ellipse
        face_angle,  # angle
        180, 0,  # start and end angles (half circle on top)
        255, -1  # color and fill
    )

    # Draw the extremum part of the polygon
    cv2.polylines(mask, [jawline], False, 255, 2)

    # Fill the mask by connecting the jawline
    # Create a closed contour including the jawline
    contour = np.vstack([jawline])
    
    cv2.fillPoly(mask, [contour], 255)
    
    # Polygon smoothing - reduced iterations
    mask = cv2.dilate(mask, SMOOTH_KERNEL)
    mask = cv2.erode(mask, SMOOTH_KERNEL)
    mask = cv2.dilate(mask, SMOOTH_KERNEL)

    return mask

def detect_and_track_faces(frame, face_cascade, img_coordinate, background):
    """Optimized face detection and tracking with landmark-based masking"""
    # Get cached backgrounds
    waiter = get_cached_background("background/waiting.jpg")
    resized_background = cv2.resize(background, BACKGROUND_SIZE)
    output = waiter
    
    # Optimize with a smaller scaling factor for detection
    scale_factor = 0.5  # Reduced from 1.0 for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Convert to grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Use faster detection with reduced parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3,  # Increased from 1.2 for faster processing
        minNeighbors=3,
        minSize=(int(30 * scale_factor), int(30 * scale_factor)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # If no faces detected, return waiting image
    if len(faces) == 0:
        return frame, output
    
    # Select only the largest face for processing (optimization)
    if len(faces) > 1:
        face_areas = [w*h for (x,y,w,h) in faces]
        largest_face_idx = np.argmax(face_areas)
        faces = [faces[largest_face_idx]]
    
    # Process the selected face
    for (x, y, w, h) in faces:
        # Scale coordinates back to original frame size
        x, y = int(x / scale_factor), int(y / scale_factor)
        w, h = int(w / scale_factor), int(h / scale_factor)
        
        # Extend face region for forehead (pre-calculate extensions)
        forehead_extension = int(h * 0.4)
        width_extension = int(w * 0.20)
        
        # Calculate extended coordinates with bounds checking
        y_extended = max(0, y - forehead_extension)
        h_extended = min(frame.shape[0] - y_extended, h + forehead_extension + (y - y_extended))
        
        x_extended = max(0, x - width_extension)
        w_extended = min(frame.shape[1] - x_extended, w + 2 * width_extension)
        
        # Extract face region with extended dimensions
        face = frame[y_extended:y_extended+h_extended, x_extended:x_extended+w_extended]
        if face.size == 0:
            continue
        
        # Convert face to grayscale for dlib (reuse gray conversion)
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect face with dlib - use faster detection
        rect = None
        dlib_faces = face_detector(gray_face, 0)  # 0 for faster detection
        if dlib_faces:
            rect = dlib_faces[0]
            
            # Create mask with rounded forehead
            face_mask = create_landmarks_mask(gray_face, rect)
            
            # Calculate output dimensions
            target_height = int(resized_background.shape[0] * img_coordinate['face_ratio'])
            aspect_ratio = w_extended / h_extended
            target_width = int(target_height * aspect_ratio)
            
            # Optimize resize operations with INTER_AREA for downsampling
            resize_method = cv2.INTER_AREA if target_width < w_extended else cv2.INTER_LINEAR
            face_resized = cv2.resize(face, (target_width, target_height), interpolation=resize_method)
            face_mask = cv2.resize(face_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            # Calculate placement (pre-compute multiplications)
            bg_x = int(img_coordinate['x_faceplacement'] * resized_background.shape[1] - target_width // 2)
            bg_y = int(img_coordinate['y_faceplacement'] * resized_background.shape[0] - target_height // 2)
            
            # Bounds checking
            if (bg_x >= 0 and bg_y >= 0 and
                bg_x + target_width <= resized_background.shape[1] and
                bg_y + target_height <= resized_background.shape[0]):
                
                # Extract region from background (use a view instead of copy when possible)
                region = resized_background[bg_y:bg_y+target_height, bg_x:bg_x+target_width]
                
                # Optimize alpha blending with vectorized operations
                # Convert mask to float32 once and normalize
                alpha = face_mask.astype(np.float32) / 255.0
                
                # Expand dimensions for broadcasting
                alpha_3ch = np.expand_dims(alpha, axis=2)
                
                # Vectorized blending
                blended = cv2.convertScaleAbs(
                    face_resized * alpha_3ch + region * (1.0 - alpha_3ch)
                )
                
                # Update output image efficiently
                output = resized_background.copy()
                output[bg_y:bg_y+target_height, bg_x:bg_x+target_width] = blended
    
    return frame, output