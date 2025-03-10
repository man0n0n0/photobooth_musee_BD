import cv2
import os
import random

#  definir les references de placement des visages pour chaque image d'arrière plan (dans le dossier background)
background_refs = {
    "join_the_team_full_hd-1.jpg" : {
    "x_faceplacement" : 0.2,
    "y_faceplacement" : 0.5,
    "face_ratio" : 0.30
    },
        "join_the_team_full_hd-2.jpg" : {
    "x_faceplacement" : 0.2,
    "y_faceplacement" : 0.5,
    "face_ratio" : 0.30
    },
        "join_the_team_full_hd-3.jpg" : {
    "x_faceplacement" : 0.2,
    "y_faceplacement" : 0.5,
    "face_ratio" : 0.30
    },
        "join_the_team_full_hd-4.jpg" : {
    "x_faceplacement" : 0.2,
    "y_faceplacement" : 0.5,
    "face_ratio" : 0.30
    },
        "join_the_team_full_hd-5.jpg" : {
    "x_faceplacement" : 0.2,
    "y_faceplacement" : 0.5,
    "face_ratio" : 0.30
    },

    "waiting.jpg" : {
    "x_faceplacement" : 0.8,
    "y_faceplacement" : 0.1,
    "face_ratio" : 0.6
    }
} 

def get_random_background(background_folder):
    """Sélectionne une image aleatoire du dossier."""
    if not os.path.exists(background_folder):
        raise FileNotFoundError(f"Dossier '{background_folder}' n'existe pas !")
    
    #exclude specific wainting image files and invalid extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    waiting_images = ['waiting.jpg','waiting.jpg']
    image_files = [
        f for f in os.listdir(background_folder)
        if os.path.splitext(f)[1].lower() in valid_extensions 
        and f not in waiting_images
    ]

    if not image_files:
        raise ValueError(f"Aucun fichier image trouvé dans '{background_folder}' !")

    background = random.choice(image_files)

    return background, background_refs[background]