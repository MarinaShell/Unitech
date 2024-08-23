import cv2
import numpy as np
import sqlite3
import time
import uuid
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from cls_base_class import FaceRecognitionBase
from time import sleep

class InsightFaceRecognition(FaceRecognitionBase):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # Инициализация ArcFace модели

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
    def add_face_to_db(self, face_image):
        face_encoding = self.encode_face(face_image)
        if len(face_encoding)>0:
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
    def encode_face(self, face_image):
        faces = self.app.get(face_image)
        # Найти самое большое лицо
        largest_face = None
        max_area = 0
        for face in faces:
            # Предполагается, что у объекта лица есть атрибуты 'bbox' (bounding box)
            # bbox = [x, y, width, height]
            x, y, w, h = face.bbox
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = face

        # Вернуть эмбеддинг самого большого лица
        if largest_face is not None:
            return largest_face.embedding
        else:
            return None

    #================================================================================
    def recognize_face(self, frame):
        guid = None
        confidence =0.0
        pr_live_person = False

        known_face_encodings, known_face_guids = self.load_known_faces()
        if not known_face_encodings or not known_face_guids:
            print("No known faces to compare.")
            return frame, guid, confidence, pr_live_person

        faces = self.app.get(frame)

        if len(faces) == 0:
            return frame, guid, confidence, pr_live_person

        largest_face = max(faces, key=lambda face: face.bbox[2] * face.bbox[3])
        x1, y1, x2, y2 = largest_face.bbox.astype(int)
        
        largest_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pr_live_person = self.livePerson(gray, largest_rect)

        face_embedding = largest_face.embedding

        guid = None
        name = "Unknown"
        for known_face_encoding, known_guid in zip(known_face_encodings, known_face_guids):
            distance = np.linalg.norm(known_face_encoding - face_embedding)
            if distance < 0.7:
                guid = known_guid
                name = "Successfully"
                confidence = 1 - distance
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

        return frame, guid, confidence, pr_live_person

    #================================================================================
    def register_user(self, cap):
        countdown = 3
        start_time = time.time()
        text = ""
        guid = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)

            faces = self.app.get(frame)
            if remaining_time <= 0:

                if len(faces) == 0:
                    text = "No face detected. Please try again."
                    break

                text, guid = self.add_face_to_db(frame)
                end_time = time.time()
                registration_time = end_time - start_time
                print(f"Registration time: {registration_time:.2f} seconds")
                break

            if len(faces) > 0:
                largest_face = max(faces, key=lambda face: face.bbox[2] * face.bbox[3])
                x1, y1, x2, y2 = largest_face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
