import cv2
import os
import random

#  definir les references de placement des visages pour chaque image d'arrière plan (dans le dossier background)
background_refs = {
    "join_the_team_full_hd-1.jpg" : {
    "x_faceplacement" : 0.33,
    "y_faceplacement" : 0.29,
    "face_ratio" : 0.35
    },
        "join_the_team_full_hd-2.jpg" : {
    "x_faceplacement" : 0.54,
    "y_faceplacement" : 0.22,
    "face_ratio" : 0.35
    },
        "join_the_team_full_hd-3.jpg" : {
    "x_faceplacement" : 0.50,
    "y_faceplacement" : 0.19,
    "face_ratio" : 0.31
    },
        "join_the_team_full_hd-4.jpg" : {
    "x_faceplacement" : 0.65,
    "y_faceplacement" : 0.22,
    "face_ratio" : 0.29
    },
        "join_the_team_full_hd-5.jpg" : {
    "x_faceplacement" : 0.45,
    "y_faceplacement" : 0.15,
    "face_ratio" : 0.4
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
