import cv2

def add_text_to_image(image, text, position=(50, 50), 
                      font_scale=1, color=(255, 0, 0), thickness=2):

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, 
                font_scale, color, thickness, cv2.LINE_AA)
    return image