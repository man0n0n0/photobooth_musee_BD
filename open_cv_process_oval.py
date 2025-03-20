import cv2
import numpy as np
import time
from functools import lru_cache

# Global constants
BACKGROUND_SIZE = (1080, 1920)

# Pre-compute common kernels
SMOOTH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
FEATHER_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

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

def create_oval_mask(face):
    """
    Create a tighter oval mask for the face.
    
    Args:
        face (numpy.ndarray): The face image
    
    Returns:
        numpy.ndarray: A binary mask with a tighter oval shape
    """
    # Create an empty mask
    h, w = face.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate the center and axes of the ellipse
    center = (w // 2, h // 2)
    # Use smaller multipliers for tighter oval 
    axes = (int(w * 0.30), int(h * 0.35))  
    
    # Draw a filled ellipse
    cv2.ellipse(
        mask,
        center,
        axes,
        0,  # angle
        0, 360,  # start and end angles (full ellipse)
        255, -1  # color and fill
    )
    
    # Create a higher resolution mask for smoother edges
    hi_res_mask = np.zeros((h*2, w*2), dtype=np.uint8)
    hi_res_center = (center[0]*2, center[1]*2)
    hi_res_axes = (axes[0]*2, axes[1]*2)
    
    # Draw high-resolution ellipse
    cv2.ellipse(
        hi_res_mask,
        hi_res_center,
        hi_res_axes,
        0,  # angle
        0, 360,  # start and end angles (full ellipse)
        255, -1  # color and fill
    )
    
    # Apply Gaussian blur for anti-aliased edges
    hi_res_mask = cv2.GaussianBlur(hi_res_mask, (15, 15), 3)
    
    # Resize back to original dimensions
    mask = cv2.resize(hi_res_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Refine the mask for better edge transitions
    # Create a feathered edge by applying additional blur
    # Reduced blur sigma for sharper transition at edges
    mask = cv2.GaussianBlur(mask, (7, 7), 1.5)
    
    return mask

def detect_and_track_faces(frame, face_cascade, img_coordinate, background):
    """Face detection and tracking with simple oval masking"""
    # Get cached backgrounds
    waiter = get_cached_background("background/waiting.jpg")
    resized_background = cv2.resize(background, BACKGROUND_SIZE)
    output = waiter
    
    # Optimize with a smaller scaling factor for detection
    scale_factor = 1  # Reduced from 1.0 for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Convert to grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Use faster detection with reduced parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(int(30 * scale_factor), int(30 * scale_factor)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # If no faces detected, return waiting image
    if len(faces) == 0:
        return frame, output, False
    
    # Select only the largest face for processing (optimization)
    if len(faces) > 1:
        face_areas = [w*h for (x,y,w,h) in faces]
        largest_face_idx = np.argmax(face_areas)
        faces = [faces[largest_face_idx]]
    
    try: 
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
            
            face_detected = True
            
            # Create oval mask for the face
            face_mask = create_oval_mask(face)
            
            # Calculate output dimensions
            target_height = int(resized_background.shape[0] * img_coordinate['face_ratio'])
            aspect_ratio = w_extended / h_extended
            target_width = int(target_height * aspect_ratio)
            
            # Use higher quality resize method for both face and mask
            face_resized = cv2.resize(face, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Use INTER_LINEAR for the mask to maintain smooth edges
            face_mask = cv2.resize(face_mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Apply additional Gaussian blur to smooth mask edges after resizing
            face_mask = cv2.GaussianBlur(face_mask, (5, 5), 1.5)
            
            # Calculate placement (pre-compute multiplications)
            bg_x = int(img_coordinate['x_faceplacement'] * resized_background.shape[1] - target_width // 2)
            bg_y = int(img_coordinate['y_faceplacement'] * resized_background.shape[0] - target_height // 2)
            
            # Bounds checking
            if (bg_x >= 0 and bg_y >= 0 and
                bg_x + target_width <= resized_background.shape[1] and
                bg_y + target_height <= resized_background.shape[0]):
                
                # Extract region from background
                region = resized_background[bg_y:bg_y+target_height, bg_x:bg_x+target_width].copy()
                
                # Optimize alpha blending with vectorized operations
                # Convert mask to float32 once and normalize
                alpha = face_mask.astype(np.float32) / 255.0
                
                # Expand dimensions for broadcasting
                alpha_3ch = np.expand_dims(alpha, axis=2)
                
                # Apply additional edge refinement for seamless blending
                # Create a slightly blurred copy of the face for edge transitions
                face_edges_blurred = cv2.GaussianBlur(face_resized, (3, 3), 0.8)
                
                # Use the blurred face for edge areas (where alpha is between 0.05 and 0.95)
                edge_mask = ((alpha_3ch > 0.05) & (alpha_3ch < 0.95))
                
                # Prepare the blended face (mix sharp interior with blurred edges)
                blended_face = face_resized.copy()
                blended_face[edge_mask.squeeze()] = face_edges_blurred[edge_mask.squeeze()]
                
                # Vectorized blending with refined face
                blended = cv2.convertScaleAbs(
                    blended_face * alpha_3ch + region * (1.0 - alpha_3ch)
                )
                
                # Update output image efficiently
                output = resized_background.copy()
                output[bg_y:bg_y+target_height, bg_x:bg_x+target_width] = blended
        
            return frame, output, face_detected

    except Exception as e:
        print(f"Error in face detection: {e}")
        return frame, output, False  # return only background if error