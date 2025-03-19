import cv2
import dlib
import numpy as np
import random

background_size = (720, 1040)

# Initialize these only once outside the functions
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def create_landmarks_mask(face, rect):
    """
    Create a mask for the face using facial landmarks from dlib,
    with a simple rounded forehead.
    
    Args:
        face (numpy.ndarray): The face image
        rect (dlib.rectangle): The face rectangle from dlib
    
    Returns:
        numpy.ndarray: A binary mask following the face contour with rounded forehead
    """
    # Create an empty mask with the same size as the face
    h, w = face.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # If no face rectangle provided, return a empty mask
    if rect is None:
        return mask
    
    # Get facial landmarks
    shape = shape_predictor(face, rect)
    points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int32)
    
    # Get jawline points
    jawline = points[0:17]

    #### ellipse hairline method
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
        
    # Polygon smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel)

    return mask

def detect_and_track_faces(frame, face_cascade, img_coordonate, background):
    """Détection et suivi des visages avec masque basé sur les points de repère faciaux et front arrondi."""
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
        
        # Extend the face region upward to include more forehead
        forehead_extension = int(h * 0.4)  # Extend by 40% of face height for rounded forehead
        y_extended = max(0, y - forehead_extension)
        h_extended = h + forehead_extension + (y - y_extended)
        
        # Extend width on both sides
        width_extension = int(w * 0.20)  # Extend by 15% on each side (30% total)
        x_extended = max(0, x - width_extension)
        w_extended = w + width_extension + (x - x_extended) + width_extension
        # Ensure we don't exceed frame boundaries
        w_extended = min(w_extended, frame.shape[1] - x_extended)
        
        # Extract face region with extended forehead and width
        face = frame[y_extended:y_extended+h_extended, x_extended:x_extended+w_extended]
        if face.size == 0:
            continue
        
        # Convert face to grayscale for dlib
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect face with dlib
        rect = None
        dlib_faces = face_detector(gray_face, 0)  # Use 0 for faster detection
        if len(dlib_faces) > 0:
            rect = dlib_faces[0]
        
        # Create mask with rounded forehead and smooth edges
        face_mask = create_landmarks_mask(gray_face, rect)
        
        # Calculate output dimensions based on target height while preserving aspect ratio
        target_height = int(resized_background.shape[0] * img_coordonate['face_ratio'])
        
        # Calculate width to maintain aspect ratio
        aspect_ratio = w_extended / h_extended
        target_width = int(target_height * aspect_ratio)
        
        # Resize face and mask while preserving aspect ratio
        face_resized = cv2.resize(face, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        face_mask = cv2.resize(face_mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate placement
        bg_x = int(img_coordonate['x_faceplacement'] * resized_background.shape[1] - target_width//2)
        bg_y = int(img_coordonate['y_faceplacement'] * resized_background.shape[0] - target_height//2)
        
        # Ensure coordinates are within bounds
        if (bg_x < 0 or bg_y < 0 or
            bg_x + target_width > resized_background.shape[1] or
            bg_y + target_height > resized_background.shape[0]):
            continue
        
        # Extract region from background
        region = resized_background[bg_y:bg_y+target_height, bg_x:bg_x+target_width].copy()
        
        # Create normalized alpha mask (0 to 1)
        alpha = face_mask.astype(float) / 255
        
        # Expand alpha to 3 channels
        alpha = cv2.merge([alpha, alpha, alpha])
        
        # Perform alpha blending: dst = alpha * src1 + (1 - alpha) * src2
        blended = cv2.convertScaleAbs(face_resized * alpha + region * (1.0 - alpha))
        
        # Update the output image
        output = resized_background.copy()
        output[bg_y:bg_y+target_height, bg_x:bg_x+target_width] = blended
    
    return frame, output