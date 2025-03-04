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

        face = frame[y:y+h, x:x+w]
        center = x + w//2, y + h//2
        
        # Création du masque elliptique au lieu du cercle
        mask = np.zeros((max(w,h),max(w,h)), dtype=np.uint8)

        # Utilisation de cv2.ellipse avec l'axe majeur=w et mineur=h/2
        cv2.ellipse(
            mask,
            (max(w,h)//2, max(w,h)//2),
            (w//3, h//2),  # Axe majeur=w/2, axe mineur=h/4
            0,  # Pas d'angle de rotation
            0,  # Angle de début
            360,  # Angle de fin
            255,  # Couleur blanche pour le masque
            -1   # Épaisseur -1 pour remplir l'ellipse
        )
        
                # Apply the mask to get elliptical face
        face_cropped = cv2.bitwise_and(face, face, mask=mask)
        
        # Redimensionnement
        resized_dia = int(output.shape[0] * img_coordonate['face_ratio'])
        face_elliptical = cv2.resize(face_cropped, (resized_dia, resized_dia), 
                                    interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (resized_dia, resized_dia))
        
        # Get smooth alpha blending for the face
        face_for_blend, alpha_mask = contour_fade(face_elliptical, mask, blur_amount=25)
        
        # Placement sur l'arrière-plan
        bg_x = int(img_coordonate['x_faceplacement'] * output.shape[1] - face_elliptical.shape[1]//2)
        bg_y = int(img_coordonate['y_faceplacement'] * output.shape[0] - face_elliptical.shape[0]//2)
        
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
     






        # face_elliptical = cv2.bitwise_and(face, face, mask=mask)
        
        # # Redimensionnement
        # resized_dia = int(output.shape[0]*img_coordonate['face_ratio'])
        # face_elliptical = cv2.resize(face_elliptical, (resized_dia, resized_dia), 
        #                    interpolation=cv2.INTER_CUBIC)
        # mask = cv2.resize(mask,(resized_dia,resized_dia))
        # contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(mask, contours, -1, (0,0,0), 1)

        # # Placement sur l'arrière-plan
        # bg_x = int(img_coordonate['x_faceplacement']*output.shape[1]-face_elliptical.shape[1]//2)
        # bg_y = int(img_coordonate['y_faceplacement']*output.shape[0]-face_elliptical.shape[0]//2)

        # # face addition to background (prone to error if face is oversized !!)
        # region = output[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])]
        # result = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(mask))
        # result = cv2.add(result, face_elliptical)
        # output[bg_y:bg_y+int(face_elliptical.shape[0]), bg_x:bg_x+int(face_elliptical.shape[1])] = result

    return frame, output