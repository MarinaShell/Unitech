import cv2
import torch
import sqlite3
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
import os

class FaceRecognitionApp:
    def __init__(self, db_path='face_mtcnn.db'):
        self.db_path = db_path
        self.create_db()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)  # Инициализация MTCNN
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)  # Инициализация Resnet


    #================================================================================    
    def create_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      encoding TEXT)''')
        conn.commit()
        conn.close()
        
    #================================================================================    
    def delete_user(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        return f"User with id {user_id} deleted successfully."

    #================================================================================    
    def show_all_users(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, name FROM faces")
        rows = c.fetchall()
        conn.close()

        if rows:
            print("Registered users:")
            for row in rows:
                print(f"ID: {row[0]}, Name: {row[1]}")
        else:
            print("No users found.")
            
    def load_known_faces(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name, encoding FROM faces")
        rows = c.fetchall()
        conn.close()

        known_face_names = [row[0] for row in rows]
        known_face_encodings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        return known_face_encodings, known_face_names
    
    def encode_face(self, face_image):
        if len(face_image)>0:
            face_image = cv2.resize(face_image, (160, 160))
            face_image = torch.tensor(face_image).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            embedding = self.resnet(face_image).detach().cpu().numpy().flatten()
            return embedding
        else:
            return None

    def add_face_to_db(self, name, face_image):
        face_encoding = self.encode_face(face_image)
        if len(face_encoding)>0:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, face_encoding.tobytes()))
            conn.commit()
            conn.close()
            return f"User {name} registered successfully."
        else:
            return "No face detected. Please try again."
        
    def recognize_face(self, frame):
        # Загрузка известных лиц и их идентификаторов
        known_face_encodings, known_face_names = self.load_known_faces()
        if not known_face_encodings or not known_face_names:
            print("No known faces to compare.")
            return frame, None

        # Используем MTCNN для детектирования лиц на фрейме
        boxes, _ = self.mtcnn.detect(frame)

        if boxes is None or len(boxes) == 0:
            return frame, None

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
            return frame, None

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
        min_distance = float('inf')

        # Сравниваем с известными лицами
        for known_face_encoding, known_name in zip(known_face_encodings, known_face_names):
            distance = np.linalg.norm(known_face_encoding - face_embedding)
            if distance < min_distance:
                min_distance = distance
                name = known_name
                  

        # Проверяем минимальное расстояние и устанавливаем имя и guid
        if min_distance >= 0.6:  # Порог для распознавания лица
            name = "Unknown"
 
        # Отображаем распознанное лицо и имя на фрейме
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

        return frame, name
    
    #================================================================================        
    def recognize_users(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            recognized_frame, names = self.recognize_face(frame_rgb)
            end_time = time.time()  # End time for recognizing this frame
            recognition_time = end_time - start_time
            print(f"Recognition time for current frame: {recognition_time:.2f} seconds")          
          
            # Display the frame
            frame_rgb = cv2.cvtColor(recognized_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

            cv2.imshow('Recognize Users', frame_rgb)            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    def register_user(self, name):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        countdown = 3
        start_time = time.time()
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
                text = self.add_face_to_db(name, largest_face_img)
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
        return text
    
        
##########################################################################
def load_images_from_folder(folder):
    images = []
    labels = []

    # Проходим по всем подкаталогам
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".bmp"):
                label = os.path.basename(subdir)  # Имя подкаталога является меткой
                img = cv2.imread(os.path.join(subdir, file))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB
                    images.append(img_rgb)
                    labels.append(label)
    
    return images, labels

##########################################################################
if __name__ == "__main__":
    app = FaceRecognitionApp()
    while True:
        choice = input("Choose an option: 1) Register User 2) Recognize Users 3) Show All Users 4) Delete User 5) Train and Evaluate Model 6) Exit\n")
        if choice == '1':
            name = input("Enter the name: ")
            app.register_user(name)
        elif choice == '2':
            app.recognize_users()
        elif choice == '3':
            app.show_all_users()
        elif choice == '4':
            user_id = input("Enter the user ID to delete: ")
            app.delete_user(user_id)
        elif choice == '5':
            train_folder = "d:/marina/new_opencv/DataSetNew/"
            test_folder = "d:/marina/new_opencv/DataSetForTest/"

            train_images, train_labels = load_images_from_folder(train_folder)
            test_images, test_labels = load_images_from_folder(test_folder)

            # Add training images to the database
            for image, label in zip(train_images, train_labels):
                  result = app.add_face_to_db(label, image)
                  print(result)

            # Evaluate the model on test images
            correct_predictions = 0
            for image, true_label in zip(test_images, test_labels):
                    recognized_frame, predicted_labels = app.recognize_face(image)
                    if true_label in predicted_labels:
                        correct_predictions += 1

            accuracy = correct_predictions / len(test_images)
            print(f"Model accuracy: {accuracy * 100:.2f}%")
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")