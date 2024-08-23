import cv2
import numpy as np
import sqlite3
from skimage.feature import local_binary_pattern
import time
import os

##########################################################################
class FaceRecognitionApp:
    def __init__(self, db_path='face_encodings_dnn.db'):
        self.db_path = db_path
        self.create_db()
        self.face_detector_model = './files/face_detection_yunet_2023mar.onnx'
        self.face_recognizer_model = './files/face_recognition_sface_2021dec.onnx'
        self.score_threshold = 0.9
        self.nms_threshold = 0.3
        self.top_k = 5000
        self.backendId = 0
        self.targetId = 0
        self._disType = 1
        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128
        self.detector = cv2.FaceDetectorYN.create(self.face_detector_model, 
                                                  "", 
                                                  (320, 320), 
                                                  self.score_threshold, 
                                                  self.nms_threshold, 
                                                  self.top_k, 
                                                  self.backendId, 
                                                  self.targetId )
        self.face_recognizer = cv2.FaceRecognizerSF.create(self.face_recognizer_model, "")
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
    def encode_face(self, face_image):
        if face_image is not None and face_image.size > 0:
            aligned_face = self.face_recognizer.alignCrop(face_image, face_image)
            face_feature = self.face_recognizer.feature(aligned_face)
            return face_feature.flatten()
        else:
            return np.array([])

    #================================================================================    
    def add_face_to_db(self, name, face_image):
        face_encoding = self.encode_face(face_image)
        face_encoding_str = ','.join(map(str, face_encoding))
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, face_encoding_str))
        conn.commit()
        conn.close()
        return f"User {name} registered successfully."

    #================================================================================    
    def load_known_faces(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name, encoding FROM faces")
        rows = c.fetchall()
        conn.close()

        known_face_encodings = [np.fromstring(row[1], sep=',') for row in rows]
        known_face_names = [row[0] for row in rows]
        return known_face_encodings, known_face_names

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
    def register_user(self, name):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        countdown = 3  # Время отсчета в секундах
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)
 
            if remaining_time > 0:
                # Отображение времени отсчета на экране
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                input_size = (frame.shape[1], frame.shape[0])
                self.detector.setInputSize(input_size)
                faces = self.detector.detect(frame)
                faces = faces[1]

                if faces is not None:
                    largest_face = None
                    max_area = 0
                    for face in faces:
                        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        area = w * h
                        if area > max_area:
                            max_area = area
                            largest_face = face

                    if largest_face is not None:
                        x, y, w, h = int(largest_face[0]), int(largest_face[1]), int(largest_face[2]), int(largest_face[3])
                        face_image = frame[y:y+h, x:x+w]
                        result = self.add_face_to_db(name, face_image)
                        end_time = time.time()
                        registration_time = end_time - start_time
                        print(result)
                        print(f"Registration time: {registration_time:.2f} seconds")
                        break
            
            cv2.imshow('Register User', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    #================================================================================    
    def register_user10(self, name):
        cap = cv2.VideoCapture(0)  # Use the first camera on the system
        if not cap.isOpened():
            print("Cannot open camera")
            return

        countdown = 5  # 5 seconds countdown
        start_time = time.time()
        frames = []
        num_frames = 10  # Number of frames to collect

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)
     
            if remaining_time > 0:
                # Отображение времени отсчета на экране
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                input_size = (frame.shape[1], frame.shape[0])
                self.detector.setInputSize(input_size)
                faces = self.detector.detect(frame)
                faces = faces[1]

                if faces is not None:
                    face = faces[0]
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    face_image = frame[y:y+h, x:x+w]
                    frames.append(face_image)
                    if len(frames) >= num_frames:
                        for i, frame in enumerate(frames):
                            result = self.add_face_to_db(name, frame)
                            print(result)
                        
                        end_time = time.time()
                        registration_time = end_time - start_time
                        print(result)
                        print(f"Registration time: {registration_time:.2f} seconds")
                        break

            # Отображение кадра
            cv2.imshow('Register User', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    #================================================================================    
    def recognize_face(self, frame):
        known_face_encodings, known_face_names = self.load_known_faces()
  
        input_size = (frame.shape[1], frame.shape[0])
        self.detector.setInputSize(input_size)
        start_time = time.time()
        faces = self.detector.detect(frame)
        faces = faces[1]
            
        name = "Unknown"
        if faces is not None:
            largest_face = None
            max_area = 0
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face = face

            min_distance = float('inf')  # Инициализируем минимальное расстояние как бесконечность

            if largest_face is not None:
                x, y, w, h = int(largest_face[0]), int(largest_face[1]), int(largest_face[2]), int(largest_face[3])
                img = frame[y:y+h, x:x+w]
                face_encoding = self.encode_face(img).astype(np.float64)
                for face_load_encoding, face_load_name in zip(known_face_encodings, known_face_names):
                    face_load_encoding = face_load_encoding.astype(np.float64)  # Приведение типов данных к float32
                    if self._disType == 0: # COSINE
                        cosine_score = self.face_recognizer.match(face_encoding, face_load_encoding, self._disType)
                        min_distance = cosine_score
                        if cosine_score < min_distance:
                            name = face_load_name
                            min_distance = cosine_score
                            #break
                    else: # NORM_L2
                        norml2_distance = self.face_recognizer.match(face_encoding, face_load_encoding, self._disType)
                        if norml2_distance < min_distance:
                            name = face_load_name
                            min_distance = norml2_distance
                            #break
                       
                    if self._disType == 0: # COSINE
                        if cosine_score < self._threshold_cosine:
                            name = "Unknown"
                    else:
                        if norml2_distance > self._threshold_norml2:
                            name =  "Unknown"
                    
                    end_time = time.time()  # End time for recognizing this frame
                    recognition_time = end_time - start_time    
                    print(f"Recognition time for current frame: {recognition_time:.2f} seconds")    
         
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)
      

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
            train_folder = "d:/marina/new_opencv/DataSet/"
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
