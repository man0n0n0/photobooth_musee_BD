# overall description
The following material is a special commition for "le musée de la bande-dessinée", brussels in the context of a early march exhibiton about the comics-streap mean of production 

# todo
- face smooth integration
- raspberrypi speed optimisation 
- opencv_process cleaning (not mandatory)

# specification
>> L'idée est que les visiteurs de l'expo prennent place sur un spot au
>> sol, et que leur visage apparaisse à l'endroit d'un dessin (et non une
>> photo comme dans le test). Le script final devrait comme évoqué tourner
>> sur Raspberry pi (ou équivalent). Une amélioration serait que le dessin
>> (et donc l'emplacement du visage) soit choisi au hasard parmi une
>> dizaine de dessins potentiel dans une liste dispo.
>>
>> L'installation sera en fin d'expo et autour d'un truc du genre "Vous
>> aussi faites partie d'une maison d'édition" avec l'insert de leur visage
>> dans un "petits métiers de l'édition" invisible pour le public, porter
>> des caisses faire des envois postaux, répondre aux mails retoucher des
>> dessins sur ordi, corriger les fautes d'orthographe, etc.
>>
>> Un truc un peu ludique et rapide à essayer, comme vous le voyez.
>>
>> Voyez si ça vous semble fastoche ou compliqué à affiner (stabilité de
>> l'image, changement d'image sans que ça saute dans tous les sens dès
>> qu'on bouge, etc.).

# Ressources
## Modèle de détection facial
- [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
- [haarcascade_frontalface_default.xml](https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
## Tuto
- [Face Swapping Open CV - Dlib](https://www.youtube.com/watch?v=dK-KxuPi768&t=260s)

# procedure of activation 

**to determine**

# bom
- raspberrypi > 3+ 
- sd-card > 16Go
- RAZER webcam 
- micro-HDMI to HDMI adapter
- **to determine** TV

# Lib

- OpenCV

```
sudo apt-get install python3-opencv
```
- numpy (automatic install as a opencv dependances)


![dafoe](https://media.giphy.com/media/d11j4By0L4cdxQbo3X/giphy.gif?cid=ecf05e47y88zsv09eweeo3my53ffhrpw8gr8f0hckf7v34o2&ep=v1_gifs_search&rid=giphy.gif&ct=g)



