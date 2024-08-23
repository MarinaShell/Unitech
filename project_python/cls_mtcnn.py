import cv2
import torch
import sqlite3
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from cls_base_class import FaceRecognitionBase
import time
import uuid

class MTCNNFaceRecognition(FaceRecognitionBase):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)  # Инициализация MTCNN
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)  # Инициализация Resnet

    #================================================================================
    def load_known_faces(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT guid, encoding FROM faces")
        rows = c.fetchall()
        conn.close()

        known_face_guids = [row[0] for row in rows]
        known_face_encodings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        return known_face_encodings, known_face_guids
    
    #================================================================================
    def encode_face(self, face_image):
        if len(face_image)>0:
            face_image = cv2.resize(face_image, (160, 160))
            face_image = torch.tensor(face_image).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            embedding = self.resnet(face_image).detach().cpu().numpy().flatten()
            return embedding
        else:
            return None

    #================================================================================
    def add_face_to_db(self, face_image):
        face_encoding = self.encode_face(face_image)
        if len(face_encoding)>0:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            guid = str(uuid.uuid4())
            c.execute("INSERT INTO faces (guid, encoding) VALUES (?, ?)", (guid, face_encoding.tobytes()))
            user_id = c.lastrowid
            conn.commit()
            conn.close()
            return f"User {user_id} registered successfully.", guid
        else:
            return "No face detected. Please try again.", None
     
    #================================================================================        
    def recognize_face(self, frame):
        guid = None
        confidence =0.0
        pr_live_person = False

        # Загрузка известных лиц и их идентификаторов
        known_face_encodings, known_face_guids = self.load_known_faces()
        if not known_face_encodings or not known_face_guids:
            print("No known faces to compare.")
            return frame, guid, confidence, pr_live_person

        # Используем MTCNN для детектирования лиц на фрейме
        boxes, _ = self.mtcnn.detect(frame)

        if boxes is None or len(boxes) == 0:
            return frame, guid, confidence, pr_live_person

        # Определяем самое большое лицо
        largest_face = None
        largest_area = 0

        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            w, h = x2 - x1, y2 - y1
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = box

        if largest_face is None:
            return frame, guid, confidence, pr_live_person

        # Получаем координаты и изображение самого большого лица
        x1, y1, x2, y2 = [int(b) for b in largest_face]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(0, x2)
        y2 = max(0, y2)
        face_img = frame[y1:y2, x1:x2]
        frame_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # Получаем эмбеддинг для самого большого лица
        face_embedding = self.encode_face(frame_rgb)

        name = "Unknown"
        guid = None
        min_distance = float('inf')

        # Сравниваем с известными лицами
        for known_face_encoding, known_guid in zip(known_face_encodings, known_face_guids):
            distance = np.linalg.norm(known_face_encoding - face_embedding)
            if distance < min_distance:
                min_distance = distance
                guid = known_guid
                  

        # Проверяем минимальное расстояние и устанавливаем имя и guid
        if min_distance >= 0.6:  # Порог для распознавания лица
            name = "Unknown"
            guid = None
        else:
            name = "Successfully"
            confidence = 1 - min_distance

        # Отображаем распознанное лицо и имя на фрейме
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, guid, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

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
        
            boxes, _ = self.mtcnn.detect(frame)
            if remaining_time <= 0:
                # Если лица найдены
                if boxes is None or len(boxes) == 0:
                    text = "No face detected. Please try again."
                    break

                # Находим самое большое лицо
                largest_face = None
                largest_area = 0

                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    w, h = x2 - x1, y2 - y1
                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest_face = box

                if largest_face is None:
                    text = "No face detected. Please try again."
                    break

                # Получаем координаты и изображение самого большого лица
                x1, y1, x2, y2 = [int(b) for b in largest_face]
                largest_face_img = frame[y1:y2, x1:x2]
                text, guid = self.add_face_to_db(largest_face_img)
                end_time = time.time()
                registration_time = end_time - start_time
                print(f"Registration time: {registration_time:.2f} seconds")
                break
                
            # Отображаем прямоугольник вокруг самого большого лица
            if boxes is not None:
                largest_face = None
                largest_area = 0
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    w, h = x2 - x1, y2 - y1
                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest_face = box
                if largest_face is not None:    
                    x1, y1, x2, y2 = [int(b) for b in largest_face]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Отображаем обратный отсчет
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Отображаем кадр
            cv2.imshow('Register User', frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Отображаем финальное сообщение
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'{text}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
   
        # Отображаем кадр
        cv2.imshow('Register User', frame)
        cap.release()
        cv2.destroyAllWindows()
        return guid