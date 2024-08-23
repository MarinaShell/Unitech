import cv2
from deepface import DeepFace
import numpy as np
import sqlite3
from cls_base_class import FaceRecognitionBase
import time
import uuid
from time import sleep

class DeepFaceRecognition(FaceRecognitionBase):
    def __init__(self, db_path):
        super().__init__(db_path)
        
    def encode_face(self, face_image):
        face_embedding = DeepFace.represent(face_image, model_name='Facenet')[0]["embedding"]
        return face_embedding
 
    #================================================================================ 
    @staticmethod
    def encode_face(face_image):
        if isinstance(face_image, str):
            face_image = cv2.imread(face_image)
        if len(face_image.shape) == 2 or face_image.shape[2] == 1:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Конвертируем изображение в RGB
        #models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace']

        face_embedding = DeepFace.represent(face_image_rgb, model_name='VGG-Face', enforce_detection= False)[0]["embedding"]
        return face_embedding

    #================================================================================
    def add_face_to_db(self, face_image):
        face_encoding = self.encode_face(face_image)
        if face_encoding:
            face_encoding_str = ','.join(map(str, face_encoding))  # Преобразование вектора в строку
            conn = sqlite3.connect(self.db_path)        
            c = conn.cursor()
            guid = str(uuid.uuid4())
            c.execute("INSERT INTO faces (guid, encoding) VALUES (?, ?)", (guid, face_encoding_str))
            user_id = c.lastrowid
            conn.commit()
            conn.close()
            return f"User {user_id} registered successfully.", guid
        else:
            return "No face detected. Please try again.", None

    #================================================================================
    def load_known_faces(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT guid, encoding FROM faces")
        rows = c.fetchall()
        conn.close()

        known_face_encodings = [np.array(eval(row[1])) for row in rows]
        known_face_guids = [row[0] for row in rows]
        return known_face_encodings, known_face_guids

    #================================================================================
    def recognize_face(self, frame):
        guid = None
        confidence =0.0
        pr_live_person = False

        known_face_encodings, known_face_guids = self.load_known_faces()
        if not known_face_encodings or not known_face_guids:
            print("No known faces to compare.")
            return frame, guid, confidence, pr_live_person

        # Use DeepFace to detect faces in the frame
        detected_faces = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)

        if len(detected_faces)==0:
            return frame, guid, confidence, pr_live_person

        # Определить самое большое лицо
        largest_face = None
        largest_area = 0

        for face in detected_faces:
            face_area = face['facial_area']
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = face

        if largest_face is None:
            return frame, guid, confidence, pr_live_person

        # Получить координаты и изображение самого большого лица
        face_area = largest_face['facial_area']
        x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
        face_img = frame[y:y+h, x:x+w]
        largest_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pr_live_person = self.livePerson(gray, largest_rect)

        # Получить эмбеддинг для самого большого лица
        face_embedding = self.encode_face(face_img)

        name = "Unknown"
        guid = None
        min_distance = float('inf') 
        for known_face_encoding, known_guid in zip(known_face_encodings, known_face_guids):
            distance = np.linalg.norm(known_face_encoding - face_embedding)
            if distance < min_distance:
                min_distance = distance
                guid = known_guid
            
         confidence = 0.0  # Низкий confidence, если лицо не распознано
        if min_distance > 0.7:  # Threshold for Facenet can be adjusted based on accuracy requirements
           name = "Unknown"
           guid = None
        else:
           name = "Successfully"
           confidence = 1 - min_distance

        # Отобразить распознанное лицо
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)

        return frame, guid, confidence, pr_live_person    
        
    #================================================================================
    def register_user(self, cap):
        countdown = 3
        start_time = time.time()
        guid = None
        text = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)
        
            detected_faces = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)
            if remaining_time <= 0:
                # Используем DeepFace для обнаружения лиц
            
                if not detected_faces:
                    text = "No face detected. Please try again."
                    break

                # Найти самое большое лицо
                largest_face = None
                largest_area = 0

                for face in detected_faces:
                    face_area = face['facial_area']
                    x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest_face = face

                if largest_face is None:
                    text = "No face detected. Please try again."
                    break

                # Получить координаты и изображение самого большого лица
                face_area = largest_face['facial_area']
                x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                largest_face_img = frame[y:y+h, x:x+w]
                frame_rgb = cv2.cvtColor(largest_face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                result, guid = self.add_face_to_db(frame_rgb)
                text = "Registration successfully"
                end_time = time.time()
                registration_time = end_time - start_time
                print(f"Registration time: {registration_time:.2f} seconds")
                break                
        
            largest_face = None
            largest_area = 0
            for face in detected_faces:
                face_area = face['facial_area']
                x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face = face
            if largest_face is not None:    
                face_area = largest_face['facial_area']
                x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display countdown
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            # Display the frame
            cv2.imshow('Register User', frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'{text}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
        # Display the frame
        cv2.imshow('Register User', frame)
        cap.release()
        cv2.destroyAllWindows()
        return guid