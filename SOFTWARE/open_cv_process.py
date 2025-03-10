import cv2
import numpy as np
import skimage.exposure

background_size = (720,1280)

# Configuration de la zone à exclure (ajouté)
EXCLUSION_ZONE = {
    'x': 100,    # Position X de la zone à exclure
    'y': 100,    # Position Y de la zone à exclure
    'w': 200,    # Largeur de la zone à exclure
    'h': 200     # Hauteur de la zone à exclure
}

def contour_fade(face, mask, blur_amount=100000):
    """
    Create a smooth fade effect around the edges of the face mask.
    
    Args:
        face: The face image
        mask: The binary mask (255 for face, 0 for background)
        blur_amount: Size of the gradient blur (higher values create wider transitions)
        
    Returns:
        face_with_alpha: Face image with a smooth alpha channel for blending
    """
    # Create a copy of the mask for blurring
    mask_blur = mask.copy()
    
    # Apply Gaussian blur to create a thin gradient
    mask_blur = cv2.GaussianBlur(mask_blur, (blur_amount, blur_amount), sigmaX=100, sigmaY=50, borderType = cv2.BORDER_CONSTANT)
   
    #Smoothing edge 
    mask_blur = skimage.exposure.rescale_intensity(mask_blur, in_range=(127.5,255), out_range=(0,255))

    # Print check
    cv2.imshow("mask",mask_blur)

    # Convert to float32 and normalize to 0-1 range for alpha blending
    alpha = mask_blur.astype(np.float32) / 255.0
    
    # Expand dimensions if needed to match face
    alpha = np.expand_dims(alpha, axis=2)
    alpha = np.repeat(alpha, 3, axis=2)
    
    # Create face with alpha channel for smooth blending
    face_rgba = face.copy().astype(np.float32)
    
    # Return face with alpha for blending
    return face_rgba, alpha

def detect_and_track_faces(frame, face_cascade, img_coordonate, background):
    """Détection et suivi des visages."""
    #output waiting background
    waiter = cv2.resize(cv2.imread(f"background/waiting.jpg"), background_size)
    output = waiter

    # Conversion en gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Création d'un masque pour exclure la zone
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    cv2.rectangle(mask, 
                 (EXCLUSION_ZONE['x'], EXCLUSION_ZONE['y']), 
                 (EXCLUSION_ZONE['x'] + EXCLUSION_ZONE['w'], 
                  EXCLUSION_ZONE['y'] + EXCLUSION_ZONE['h']), 
                 0, -1)

    # Appliquer le masque à l'image en gris
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Détecter les visages sur l'image masquée
    #faces = face_cascade.detectMultiScale(gray_masked, 1.3, 5)
    
    # DONT use the exlusion zone (super lacky on manono setup)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        output = cv2.resize(background, background_size)

        # MODIFIED: Expand the face region to include more of the hair and chin
        # Calculate expanded face boundaries with proper boundary checking
        expand_ratio_x = 0.5  # Expand width by 50%
        expand_ratio_y = 0.5  # Expand height by 50%
        
        new_w = int(w * (1 + expand_ratio_x))
        new_h = int(h * (1 + expand_ratio_y))
        
        # Calculate new top-left corner to keep the face centered
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        
        # Make sure we don't go beyond frame boundaries
        new_w = min(new_w, frame.shape[1] - new_x)
        new_h = min(new_h, frame.shape[0] - new_y)
        
        # Extract the expanded face region
        face = frame[new_y:new_y+new_h, new_x:new_x+new_w]
        
        # FIXED: Create a mask with the exact same dimensions as the face
        # Instead of using max(new_w,new_h), create a mask with the exact face dimensions
        mask = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # Calculate the ellipse center and size based on actual face dimensions
        center_x, center_y = new_w // 2, new_h // 2
        
        # MODIFIED: Make the ellipse larger to better encompass hair and chin
        # Draw ellipse directly with the face dimensions
        cv2.ellipse(
            mask,
            (center_x, center_y),  # Center of the ellipse matches center of face
            (int(new_w * 0.37), int(new_h * 0.5)),  # Better cover hair and chin
            0,  # No rotation
            0,  # Start angle
            360,  # End angle
            255,  # White color for mask
            -1   # Fill the ellipse
        )
        
        # Display the mask (for debugging)
        cv2.imshow("m", mask)
        
        # FIXED: Now the mask has the same dimensions as the face, so bitwise_and will work
        face_cropped = cv2.bitwise_and(face, face, mask=mask)
        
        # Redimensionnement
        resized_dia = int(output.shape[0] * img_coordonate['face_ratio'])
        
        # Resize both face and mask to the same dimensions
        face_elliptical = cv2.resize(face_cropped, (resized_dia, resized_dia), 
                                    interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(mask, (resized_dia, resized_dia))
        
        # MODIFIED: Increased blur amount for smoother transition at the edges
        # Get smooth alpha blending for the face
        face_for_blend, alpha_mask = contour_fade(face_elliptical, mask_resized, blur_amount=35)
        
        # Placement sur l'arrière-plan
        bg_x = int(img_coordonate['x_faceplacement'] * output.shape[1] - face_elliptical.shape[1]//2)
        bg_y = int(img_coordonate['y_faceplacement'] * output.shape[0] - face_elliptical.shape[0]//2)
        
        # Make sure the coordinates are valid
        if bg_x < 0 or bg_y < 0 or bg_x + resized_dia > output.shape[1] or bg_y + resized_dia > output.shape[0]:
            continue
        
        # Extract the region for blending
        region = output[bg_y:bg_y+face_elliptical.shape[0], bg_x:bg_x+face_elliptical.shape[1]]
        
        # Skip if region is outside the frame
        if region.shape[:2] != face_elliptical.shape[:2]:
            continue
            
        # Convert region to float for blending
        region_float = region.astype(np.float32)
        
        # Alpha blending: result = alpha*face + (1-alpha)*background
        blended = cv2.multiply(alpha_mask, face_for_blend) + cv2.multiply(1.0 - alpha_mask, region_float)
        
        # Convert back to uint8
        blended = blended.astype(np.uint8)
        
        # Place the blended result back into the output
        output[bg_y:bg_y+face_elliptical.shape[0], bg_x:bg_x+face_elliptical.shape[1]] = blended
     
    return frame, output