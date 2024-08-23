import cv2
import os
import numpy as np
import sqlite3
import time
import shutil

class FaceRecognitionApp:
    def __init__(self, db_path='face_encodings_opencv.db', training_data_path='training-data'):
        self.db_path = db_path
        self.training_data_path = training_data_path
        self.face_cascade = cv2.CascadeClassifier('./files/haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.user_id = -1
        self.labels = {}
        self.create_db()
        self.load_known_faces()

    def create_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT)''')
        conn.commit()
        conn.close()

    def load_known_faces(self):
        if os.path.exists('lbph_model.yml'):
            self.face_recognizer.read('lbph_model.yml')
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT id, name FROM faces")
            rows = c.fetchall()
            conn.close()
            self.labels = {row[0]: row[1] for row in rows}

    def prepare_training_data(self):
        faces = []
        labels = []
        for subdir, _, files in os.walk(self.training_data_path):
            label = os.path.basename(subdir)
            for file in files:
                if file.endswith(".jpg") or file.endswith(".bmp"):
                    image_path = os.path.join(subdir, file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    faces_rects = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

                    for (x, y, w, h) in faces_rects:
                        face = image[y:y+w, x:x+h]
                        faces.append(face)
                        labels.append(int(label))

        return faces, labels

    def train_lbph_model(self):
        faces, labels = self.prepare_training_data()

        if os.path.exists('lbph_model.yml'):
            self.face_recognizer.read('lbph_model.yml')
            self.face_recognizer.update(faces, np.array(labels))
        else:
            self.face_recognizer.train(faces, np.array(labels))

        self.face_recognizer.save('lbph_model.yml')

    def register_user(self, name, image=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO faces (name) VALUES (?)", (name,))
        self.user_id = c.lastrowid
        conn.commit()
        conn.close()

        user_folder = os.path.join(self.training_data_path, str(self.user_id))
        os.makedirs(user_folder, exist_ok=True)

        if image is None:
            cap = cv2.VideoCapture(0)
            count = 0
            start_time = time.time()

            while cap.isOpened() and count < 10:
                ret, frame = cap.read()
                if not ret:
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_rects = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces_rects:
                    face = gray_frame[y:y+w, x:x+h]
                    face_image_path = os.path.join(user_folder, f"{name}_{count}.jpg")
                    cv2.imwrite(face_image_path, face)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Capturing {count}/10', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                count += 1

                cv2.imshow('Register User', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_rects = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            count = 0
            for (x, y, w, h) in faces_rects:
                face = gray_frame[y:y+w, x:x+h]
                face_image_path = os.path.join(user_folder, f"{name}_{count}.jpg")
                cv2.imwrite(face_image_path, face)
                count += 1

        self.train_lbph_model()

        shutil.rmtree(user_folder)

        if image is None:
            end_time = time.time()
            registration_time = end_time - start_time
            print(f"User {name} registered successfully in {registration_time:.2f} seconds")
        else:
            print(f"User {name} registered successfully")

    def recognize_face(self, frame):
        self.load_known_faces()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        label_text = "Unknown"
        for (x, y, w, h) in faces_rects:
            face = gray_frame[y:y+w, x:x+h]
            label, confidence = self.face_recognizer.predict(face)
            label_text = self.labels.get(label, "Unknown")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label_text} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame, label_text

    def recognize_users(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            recognized_frame, names = self.recognize_face(frame_rgb)
            end_time = time.time()
            recognition_time = end_time - start_time
            print(f"Recognition time for current frame: {recognition_time:.2f} seconds")

            frame_rgb = cv2.cvtColor(recognized_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Recognize Users', frame_rgb)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

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

    def delete_user(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        print(f"User with ID {user_id} has been deleted.")

def load_images_from_folder(folder):
    images = []
    labels = []

    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".bmp"):
                label = os.path.basename(subdir)
                img = cv2.imread(os.path.join(subdir, file))
                if img is not None:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(img_gray)
                    labels.append(label)

    return images, labels

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
                app.register_user(label, image)

            # Evaluate the model on test images
            correct_predictions = 0
            for image, true_label in zip(test_images, test_labels):
                recognized_frame, predicted_label = app.recognize_face(image)
                if true_label == predicted_label:
                    correct_predictions += 1

            accuracy = correct_predictions / len(test_images)
            print(f"Model accuracy: {accuracy * 100:.2f}%")
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")