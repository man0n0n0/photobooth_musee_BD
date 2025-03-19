import cv2
import numpy as np

# Chargement de l'image
image = cv2.imread('dot.png')

# Application du flou gaussien
blurred_img = cv2.GaussianBlur(image, (21, 21), 0)

# Création du masque
mask = np.zeros(image.shape, np.uint8)

# Conversion en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Seuillage (notez l'index [1] au lieu de [2])
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

# Détection des contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dessin des contours sur le masque
cv2.drawContours(mask, contours, -1, (255,255,255), 5)

# Application du masque pour combiner les images
output = np.where(mask==np.array([255, 255, 255]), blurred_img, image)

# Affichage des résultats
cv2.imshow('Original', image)
cv2.imshow('Flou', blurred_img)
cv2.imshow('Masque', mask)
cv2.imshow('Résultat', output)
cv2.waitKey(0)
cv2.destroyAllWindows()