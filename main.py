from flask import Flask, Response, render_template
import cv2
import numpy as np
import os
import time
import logging
from datetime import datetime

# Configuration du fichier de logs
log_filename = f"./admin/logs-{datetime.now().strftime('%d-%m-%Y')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialiser la caméra
cam = cv2.VideoCapture(0)

# Définir la fréquence d'images souhaitée
TARGET_FPS = 10000

# Paramètres de l'entraînement
MIN_SAMPLES = 20
img_folder_path = 'img'
nop_folder_path = './nop'  # Dossier contenant les images sans visages
capture_folder_path = './captures'  # Dossier pour enregistrer les captures
unknown_folder_path = './a-traiter'  # Dossier pour les visages inconnus
os.makedirs(capture_folder_path, exist_ok=True)
os.makedirs(unknown_folder_path, exist_ok=True)

# Charger les modèles 3D de détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_3d_model = cv2.face.LBPHFaceRecognizer_create()
model = cv2.face.LBPHFaceRecognizer_create()

images = []
labels = []
label_dict = {}
next_label = 0
nop_images = set()

def charger_nop_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Erreur de chargement de l'image : {image_path}")
                continue
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) == 0:  # Pas de visage détecté
                nop_images.add(image_path)
                logging.info(f"Image sans visage ajoutée à la liste 'nop' : {image_path}")
            else:
                logging.info(f"Image ignorée dans le dossier /nop car des visages ont été détectés : {image_path}")

def charger_images_du_repertoire(folder_path):
    global next_label
    for person_folder in os.listdir(folder_path):
        person_folder_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            label = person_folder
            if label not in label_dict:
                label_dict[label] = next_label
                next_label += 1
            logging.info(f"Traitement du dossier: {label}")
            for filename in os.listdir(person_folder_path):
                if filename.lower().endswith((".jpg", ".png")):
                    image_path = os.path.join(person_folder_path, filename)
                    if not os.path.exists(image_path):
                        logging.warning(f"Fichier non trouvé : {image_path}")
                        continue
                    if image_path in nop_images:
                        logging.info(f"Image ignorée car présente dans /nop : {image_path}")
                        continue
                    img = cv2.imread(image_path)
                    if img is None:
                        logging.error(f"Erreur de chargement de l'image : {image_path}")
                        continue
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                    for (x, y, w, h) in faces:
                        roi = gray_img[y:y+h, x:x+w]
                        images.append(roi)
                        labels.append(label_dict[label])
                        logging.info(f"Image traitée : {image_path}, label : {label}")

# Charger les images du dossier /nop et les marquer
charger_nop_images(nop_folder_path)

# Charger les autres images à entraîner
charger_images_du_repertoire(img_folder_path)

if len(images) >= MIN_SAMPLES:
    model.train(images, np.array(labels))
    logging.info("Modèle entraîné avec succès.")
else:
    logging.warning("Pas assez d'images pour entraîner le modèle.")

def generate_frames():
    prev_time = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            logging.error("Erreur lors de la capture de l'image depuis la caméra.")
            break
        
        # Calculer le temps écoulé et ajuster le délai pour atteindre environ 60 FPS
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        sleep_time = max(1.0 / TARGET_FPS - elapsed_time, 0)
        time.sleep(sleep_time)
        prev_time = curr_time
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            face_label = "Inconnu"
            if len(images) >= MIN_SAMPLES:
                label, confidence = model.predict(roi)
                if confidence < 100:
                    face_label = [k for k, v in label_dict.items() if v == label][0]
                else:
                    face_label = "Inconnu"
                    # Enregistrer l'image du visage inconnu dans le dossier /a-traiter
                    unknown_filename = f"{unknown_folder_path}/unknown_{int(curr_time)}.jpg"
                    cv2.imwrite(unknown_filename, roi)
                    logging.info(f"Visage inconnu détecté et enregistré : {unknown_filename}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, face_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            logging.info(f"Visage détecté: Label={face_label}, Confiance={confidence}")

        # Capture et enregistre une image toutes les 2 secondes
        if int(curr_time) % 2 == 0:  
            capture_filename = f"{capture_folder_path}/capture_{int(curr_time)}.jpg"
            cv2.imwrite(capture_filename, frame)
            logging.info(f"Image capturée et enregistrée : {capture_filename}")

        # Convertir le cadre en JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
