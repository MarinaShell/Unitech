import cv2
import numpy as np
import sqlite3
import time
import uuid
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from cls_base_class import FaceRecognitionBase
from time import sleep

class FaceRecognitionApp:
    def __init__(self, db_path='face_mtcnn.db'):
    self.db_path = db_path
    self.create_db()
    self.app = FaceAnalysis()
    self.app.prepare(ctx_id = 0, det_size = (640, 640))  # Инициализация ArcFace модели

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

    #================================================================================
    def load_known_faces(self) :
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name, encoding FROM faces")
        rows = c.fetchall()
        conn.close()
        known_face_encodings = [np.array(eval(row[1])) for row in rows]
        known_face_names = [row[0] for row in rows]
        return known_face_encodings, known_face_names

    #================================================================================
    def add_face_to_db(self, name, face_image) :
        face_encoding = self.encode_face(face_image)
        if len(face_encoding) > 0:
            face_encoding_str = ','.join(map(str, face_encoding))  # Преобразование вектора в строку
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, face_encoding_str))
            conn.commit()
            conn.close()
            return f"User {name} registered successfully."
        else:
            return "No face detected. Please try again."

#================================================================================
    def encode_face(self, face_image) :
        faces = self.app.get(face_image)
        # Найти самое большое лицо
        largest_face = None
        max_area = 0
        for face in faces :
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
    def recognize_face(self, frame) :
        known_face_encodings, known_face_names = self.load_known_faces()
        if not known_face_encodings or not known_face_names :
            print("No known faces to compare.")
            return frame, None

        faces = self.app.get(frame)

        if len(faces) == 0:
            return frame, None

        largest_face = max(faces, key = lambda face : face.bbox[2] * face.bbox[3])
        x1, y1, x2, y2 = largest_face.bbox.astype(int)

        face_embedding = largest_face.embedding

        guid = None
        name = "Unknown"
        for known_face_encoding, known_name in zip(known_face_encodings, known_face_names) :
            distance = np.linalg.norm(known_face_encoding - face_embedding)
            if distance < 0.7 :
                name = known_name
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

        return frame, guid

    #================================================================================
    def register_user(self, name, cap) :
        countdown = 3
        start_time = time.time()
        text = ""
        guid = None

        while cap.isOpened() :
            ret, frame = cap.read()
            if not ret :
                break

            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)

            faces = self.app.get(frame)
            if remaining_time <= 0:
                if len(faces) == 0 :
                    text = "No face detected. Please try again."
                    break

                text = self.add_face_to_db(name, frame)
                end_time = time.time()
                registration_time = end_time - start_time
                print(f"Registration time: {registration_time:.2f} seconds")
                break

            if len(faces) > 0:
                largest_face = max(faces, key = lambda face : face.bbox[2] * face.bbox[3])
                x1, y1, x2, y2 = largest_face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Register User', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') :
                break

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'{text}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Register User', frame)
        cap.release()
        cv2.destroyAllWindows()
        return guid

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