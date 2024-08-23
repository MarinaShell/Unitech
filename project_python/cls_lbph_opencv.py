import cv2
import numpy as np
import os
import sqlite3
import time
import shutil
from cls_base_class import FaceRecognitionBase
import uuid
from time import sleep

class LBPHFaceRecognition(FaceRecognitionBase):
    def __init__(self, db_path, training_data_path='training-data'):
        super().__init__(db_path)
        self.face_cascade = cv2.CascadeClassifier('./files/haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.training_data_path = training_data_path
        self.load_known_faces()

    #================================================================================ 
    def encode_face(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        return gray

    #================================================================================
    def train_lbph_model(self):
        faces, labels = self.prepare_training_data()
        if os.path.exists('lbph_model.yml'):
            self.face_recognizer.read('lbph_model.yml')
            self.face_recognizer.update(faces, np.array(labels))
        else:
            self.face_recognizer.train(faces, np.array(labels))
        self.face_recognizer.save('lbph_model.yml')

    #================================================================================
    def load_known_faces(self):
        if os.path.exists('lbph_model.yml'):
            self.face_recognizer.read('lbph_model.yml')
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT id, guid FROM faces")  
            rows = c.fetchall()
            conn.close()
            self.labels = {row[0]: row[1] for row in rows} 
 
    #================================================================================
    def prepare_training_data(self):
        faces = []
        labels = []
        subject_dir_path = os.path.join(self.training_data_path, str(self.user_id))
        if os.path.exists(subject_dir_path):
            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:
                image_path = os.path.join(subject_dir_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces_rects = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces_rects:
                    face = image[y:y+w, x:x+h]
                    faces.append(face)
                    labels.append(self.user_id)

        return faces, labels

    #================================================================================  
    def recognize_face(self, frame):
        guid = None
        confidence =0.0
        pr_live_person = False

        self.load_known_faces()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = self.face_cascade.detectMultiScale(gray_frame,
                                                         scaleFactor=1.1,
                                                         minNeighbors=3,
                                                         minSize=(200, 300))  # минимальный размер лица
        if len(faces_rects) == 0:
            return frame, guid, confidence, pr_live_person

        # ќпределить самое большое лицо
        largest_face = None
        largest_area = 0

        for (x, y, w, h) in faces_rects:
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)

        if largest_face is None:
            return frame, guid, confidence, pr_live_person

        x, y, w, h = largest_face
        face = gray_frame[y:y+h, x:x+w]
        label, confidence = self.face_recognizer.predict(face)
        guid = self.labels.get(label, None)

        largest_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pr_live_person = self.livePerson(gray, largest_rect)

        # ќтобразить распознанное лицо
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame, guid, confidence, pr_live_person

    #================================================================================
    def add_face_to_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        guid = str(uuid.uuid4())   
        c.execute("INSERT INTO faces (guid) VALUES (?)", (guid,))
        self.user_id = c.lastrowid
        conn.commit()
        conn.close()
        return guid
    
    #================================================================================
    def register_user(self, cap):
        guid = self.add_face_to_db()
        
        user_folder = os.path.join(self.training_data_path, str(self.user_id))
        os.makedirs(user_folder, exist_ok=True)

        count = 0
        start_time = time.time()                 

        while cap.isOpened() and count < 10:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rects = self.face_cascade.detectMultiScale(gray_frame,
                                                                scaleFactor=1.1, 
                                                                minNeighbors=3, 
                                                                minSize=(200, 300))  # минимальный размер лица

            if len(faces_rects) == 0:
                continue

            # Ќайти самое большое лицо
            largest_face = None
            largest_area = 0

            for (x, y, w, h) in faces_rects:
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, w, h)

            if largest_face is not None:
                x, y, w, h = largest_face
                face = gray_frame[y:y+h, x:x+w]
                face_image_path = os.path.join(user_folder, f"{self.user_id}_{count}.jpg")
                cv2.imwrite(face_image_path, face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Capturing {count}/10', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                count += 1

            # Display the frame
            cv2.imshow('Register User', frame)
  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.train_lbph_model()

        shutil.rmtree(user_folder)

        end_time = time.time()
        registration_time = end_time - start_time
        print(f"User {self.user_id} registered successfully in {registration_time:.2f} seconds")
        
        text = "Registration successfully"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'{text}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
        # Display the frame
        cv2.imshow('Register User', frame)
        cap.release()
        cv2.destroyAllWindows()
        return guid


  