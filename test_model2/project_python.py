import cv2
import face_recognition
import time
import numpy as np
import sqlite3
from skimage.feature import local_binary_pattern
from scipy.spatial import distance as dist
import dlib
import os

##########################################################################
class FaceRecognitionApp:
    def __init__(self, db_path='face_encodings.db'):
        self.db_path = db_path
        self.create_db()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./files/shape_predictor_68_face_landmarks.dat")
        self.EAR_THRESHOLD = 0.2
        self.EAR_CONSEC_FRAMES = 3  # Number of consecutive frames the eye must be below the threshold
        self.blink_sequence = []
        
    #================================================================================
    def create_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      encoding BLOB)''')
        conn.commit()
        conn.close()

    #================================================================================
    @staticmethod
    def encode_face(face_encoding):
        return face_encoding.tobytes()

    #================================================================================
    @staticmethod
    def decode_face(binary):
        return np.frombuffer(binary, dtype=np.float64)

    #================================================================================
    def add_face_to_db(self, name, frame):
        face_encodings = face_recognition.face_encodings(frame)
        if face_encodings:
            face_encoding = face_encodings[0]
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, self.encode_face(face_encoding)))
            conn.commit()
            conn.close()
            return f"User {name} registered successfully."
        else:
            return "No face detected. Please try again."

    #================================================================================
    def load_known_faces(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name, encoding FROM faces")
        rows = c.fetchall()
        conn.close()

        known_face_encodings = [self.decode_face(row[1]) for row in rows]
        known_face_names = [row[0] for row in rows]
        return known_face_encodings, known_face_names

    ##########################################################################
    # def eye_aspect_ratio(self, eye):
    #     # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y) - coordinates
    #     A = dist.euclidean(eye[1], eye[5])
    #     B = dist.euclidean(eye[2], eye[4])

    #     # Compute the euclidean distance between the horizontal eye landmark (x, y) - coordinates
    #     C = dist.euclidean(eye[0], eye[3])

    #     # Compute the eye aspect ratio
    #     ear = (A + B) / (2.0 * C)

    #     return ear

    # ##########################################################################
    # def detect_blink(self, shape):
    #     left_eye = shape[36:42]
    #     right_eye = shape[42:48]

    #     leftEAR = self.eye_aspect_ratio(left_eye)
    #     rightEAR = self.eye_aspect_ratio(right_eye)

    #     ear = (leftEAR + rightEAR) / 2.0

    #     return ear < 0.2

    #================================================================================
    @staticmethod
    def analyze_skin_texture(face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist

    #================================================================================
    def is_real_face(self, face_image):
        hist = self.analyze_skin_texture(face_image)
        # «десь можно определить порог или использовать машинное обучение дл€ классификации текстуры
        # ƒл€ простоты возвращаем True, предполага€, что лицо насто€щее.
        # –еализаци€ может быть усложнена с использованием заранее обученной модели
        return np.sum(hist) > 0.5  # ѕростой порог дл€ демонстрации

    #================================================================================
    def normalize_face(self, shape, size=256):
        left = min([p[0] for p in shape])
        right = max([p[0] for p in shape])
        top = min([p[1] for p in shape])
        bottom = max([p[1] for p in shape])
        
        face_width = right - left
        face_height = bottom - top
        
        scale = size / max(face_width, face_height)
        
        normalized_shape = [(int(p[0] * scale), int(p[1] * scale)) for p in shape]
        return normalized_shape

    #================================================================================
    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y) - coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal eye landmark (x, y) - coordinates
        C = dist.euclidean(eye[0], eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear

    #================================================================================
    def detect_blink(self, shape):
        normalized_shape = self.normalize_face(shape)
        
        left_eye = normalized_shape[36:42]
        right_eye = normalized_shape[42:48]
        
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        leftEAR = self.eye_aspect_ratio(left_eye)
        rightEAR = self.eye_aspect_ratio(right_eye)

        ear = (leftEAR + rightEAR) / 2.0
        if ear < self.EAR_THRESHOLD:
            if not self.blink_sequence or self.blink_sequence[-1] == 'open':
                self.blink_sequence.append('closed')
        else:
            if self.blink_sequence and self.blink_sequence[-1] == 'closed':
                self.blink_sequence.append('open')
                if len(self.blink_sequence) >= 3 and self.blink_sequence[-3:] == ['open', 'closed', 'open']:
                    self.blink_sequence = []  # Reset sequence after a complete blink
                    return True
        return False

    #================================================================================
    def detect_specular_reflection(self, eye_region):
        # ѕреобразовать область глаза в градации серого
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        # ѕрименить пороговое значение дл€ обнаружени€ €рких пикселей
        _, threshold_eye = cv2.threshold(gray_eye, 20, 255, cv2.THRESH_BINARY)
        # –ассчитать процент €рких пикселей (специкул€рных отражений)
        reflection_percentage = np.sum(threshold_eye == 255) / threshold_eye.size
        return reflection_percentage > 0.05  # ѕороговое значение можно настроить

    #================================================================================
    def analyze_reflections(self, shape, frame):
        # »звлечение областей глаз из кадра
        left_eye_region = frame[shape[37][1]:shape[41][1], shape[36][0]:shape[39][0]]
        right_eye_region = frame[shape[43][1]:shape[47][1], shape[42][0]:shape[45][0]]
        # јнализ специкул€рных отражений в област€х глаз
        left_eye_reflection = self.detect_specular_reflection(left_eye_region)
        right_eye_reflection = self.detect_specular_reflection(right_eye_region)
        return left_eye_reflection and right_eye_reflection   

    #================================================================================
    def recognize_face(self, frame):
        known_face_encodings, known_face_names = self.load_known_faces()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        recognized_names = []
        pr_blink = False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for face_encoding, face_location, rect in zip(face_encodings, face_locations, rects):
            shape = self.predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                recognized_names.append(name)
            
            if self.detect_blink(shape): #and self.analyze_reflections(shape, frame):
                pr_blink = True
            else:
                pr_blink = False

        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame, recognized_names, pr_blink

    #================================================================================
    def recognize_users(self):
        cap = cv2.VideoCapture(0)
        count = 0
        name = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()  # Start time for recognizing this frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            recognized_frame, names, pr_blink = self.recognize_face(frame_rgb)
            end_time = time.time()  # End time for recognizing this frame
            recognition_time = end_time - start_time
            print(f"Recognition time for current frame: {recognition_time:.2f} seconds")          

            if (pr_blink and count<100):
                name = "real person"
                count = 0
            else:
                name =""

            count = count + 1
            if (count > 100):
                count = 0
                
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(recognized_frame, name, (20, 20), font, 1.0, (255, 255, 255), 1)


            # Display the frame
            frame_rgb = cv2.cvtColor(recognized_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            cv2.imshow('Recognize Users', frame_rgb);#recognized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()        

            
    #================================================================================
    def register_user(self, name):
        cap = cv2.VideoCapture(0)
        countdown = 3
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)
            
            if remaining_time <= 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                result = self.add_face_to_db(name, frame_rgb)
                end_time = time.time()
                registration_time = end_time - start_time
                print(result)
                print(f"Registration time: {registration_time:.2f} seconds")
                cap.release()
                cv2.destroyAllWindows()
                return
            
            # Detect face
            face_locations = face_recognition.face_locations(frame)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Display countdown
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow('Register User', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
                    
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
    def delete_user(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        print(f"User with ID {user_id} has been deleted.")

##########################################################################
def load_images_from_folder(folder):
    images = []
    labels = []

    # ѕроходим по всем подкаталогам
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".bmp"):
                label = os.path.basename(subdir)  # »м€ подкаталога €вл€етс€ меткой
                img = cv2.imread(os.path.join(subdir, file))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ѕреобразуем в RGB
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
                    recognized_frame, predicted_labels, pr_blink = app.recognize_face(image)
                    if true_label in predicted_labels:
                        correct_predictions += 1

            accuracy = correct_predictions / len(test_images)
            print(f"Model accuracy: {accuracy * 100:.2f}%")
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")
 