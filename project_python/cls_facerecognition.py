import cv2
import face_recognition
import time
import numpy as np
import sqlite3
from skimage.feature import local_binary_pattern
from scipy.spatial import distance as dist
import dlib
from cls_base_class import FaceRecognitionBase
import uuid
from time import sleep

##########################################################################
class FaceRecognition(FaceRecognitionBase):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.detector = dlib.get_frontal_face_detector()

    #================================================================================
    def encode_face(self, face_encoding):
        return face_encoding.tobytes()

    #================================================================================
    @staticmethod
    def decode_face(binary):
        return np.frombuffer(binary, dtype=np.float64)

    #================================================================================
    def load_known_faces(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT guid, encoding FROM faces")
        rows = c.fetchall()
        conn.close()
        known_face_encodings = [self.decode_face(row[1]) for row in rows]
        known_face_guids = [row[0] for row in rows]
        return known_face_encodings, known_face_guids
   
    #================================================================================
    def add_face_to_db(self, frame):
        face_encodings = face_recognition.face_encodings(frame)
        if face_encodings:
            face_encoding = face_encodings[0]
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            guid = str(uuid.uuid4())
            c.execute("INSERT INTO faces (guid, encoding) VALUES (?, ?)", (guid, self.encode_face(face_encoding)))
            user_id = c.lastrowid
            conn.commit()
            conn.close()
            return f"User {user_id} registered successfully.", guid
        else:
            return f"No face detected. Please try again.", None
    
    #================================================================================
    def recognize_face(self, frame):      
        guid = None
        confidence =0.0
        pr_live_person = False

        known_face_encodings, known_face_guids = self.load_known_faces()
        if not known_face_encodings or not known_face_guids:
            print("No known faces to compare.")
            return frame, guid, confidence, pr_live_person
    
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations)==0:
            return frame, guid, confidence, pr_live_person
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if len(rects) == 0:
            return frame, guid, confidence, pr_live_person

        # ќпределить самое большое лицо
        largest_face_index = -1
        max_area = 0
        i = 0
        largest_rect = None
        for rect in rects:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            area = w * h
            if area > max_area:
                max_area = area
                largest_face_index = i
                largest_rect = rect
            i=i+1    

        if largest_face_index==-1:
            return frame, guid, confidence, pr_live_person

        pr_live_person = self.livePerson(gray, largest_rect)

        # ѕолучить соответствующие координаты и кодировку дл€ самого большого лица
        largest_face_location = face_locations[largest_face_index]
        largest_face_encoding = face_encodings[largest_face_index]

        # —равнить с известными лицами
        matches = face_recognition.compare_faces(known_face_encodings, largest_face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, largest_face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            guid = known_face_guids[best_match_index]
            name = guid
            confidence = 1.0 - face_distances[best_match_index]

        # ќтобразить распознанное лицо
        top, right, bottom, left = largest_face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({confidence*100:.2f}%)", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame, guid, confidence, pr_live_person
            
    #================================================================================
    def register_user(self, cap):
        countdown = 3
        start_time = time.time()
        text = ""
        guid = None
        font = cv2.FONT_HERSHEY_SIMPLEX

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            face_locations = face_recognition.face_locations(gray_frame)
  
            if remaining_time <= 0:
                if not face_locations:
                    text = "No face detected. Please try again."
                    break
            
                # Ќайти самое большое лицо
                largest_face = None
                largest_area = 0

                for (top, right, bottom, left) in face_locations:
                    width = right - left
                    height = bottom - top
                    area = width * height

                    if area > largest_area:
                        largest_area = area
                        largest_face = (top, right, bottom, left)

                if largest_face is None:
                    text = "No face detected. Please try again."
                    break

                top, right, bottom, left = largest_face
                largest_face_img = frame[top:bottom, left:right]
                frame_rgb = cv2.cvtColor(largest_face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                text, guid  = self.add_face_to_db(frame_rgb)
                end_time = time.time()
                registration_time = end_time - start_time
                print(f"Registration time: {registration_time:.2f} seconds")
                break
        
            # Detect face
            
            if not face_locations:
                text = "No face detected. Please try again."
                cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                largest_area = 0
                largest_face = None
                for (top, right, bottom, left) in face_locations:
                     width = right - left
                     height = bottom - top
                     area = width * height

                     if area > largest_area:
                            largest_area = area
                            largest_face = (top, right, bottom, left)
            
                if largest_face is not None:    
                    top, right, bottom, left = largest_face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)        
                    # Display countdown
                cv2.putText(frame, f'Registering in {remaining_time}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            # Display the frame
            cv2.imshow('Register User', frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
        # Display the frame
        cv2.imshow('Register User', frame)
        sleep(5)
        cap.release()
        cv2.destroyAllWindows()
        return guid