import cv2
import numpy as np
import sqlite3
from skimage.feature import local_binary_pattern
import time
from cls_base_class import FaceRecognitionBase
import uuid
from time import sleep

##########################################################################
class DNNRecognition(FaceRecognitionBase):
    def __init__(self, db_path):
        super().__init__(db_path)
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
    def encode_face(self, face_image):
        if face_image is not None and face_image.size > 0:
            aligned_face = self.face_recognizer.alignCrop(face_image, face_image)
            face_feature = self.face_recognizer.feature(aligned_face)
            return face_feature.flatten()
        else:
            return np.array([])

    #================================================================================    
    def add_face_to_db(self, face_image):
        face_encoding = self.encode_face(face_image)
        if face_encoding:
            face_encoding_str = ','.join(map(str, face_encoding))
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

        known_face_encodings = [np.fromstring(row[1], sep=',') for row in rows]
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

        input_size = (frame.shape[1], frame.shape[0])
        self.detector.setInputSize(input_size)
        start_time = time.time()
        faces = self.detector.detect(frame)
        faces = faces[1]
            
        if faces is None:
            return frame, guid, confidence, pr_live_person
        
        name = "Unknown"
        largest_face = None
        max_area = 0
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = face

        if largest_face is None:
            return frame, guid, confidence, pr_live_person

        min_distance = float('inf')  # Инициализируем минимальное расстояние как бесконечность
        guid = None
        x, y, w, h = int(largest_face[0]), int(largest_face[1]), int(largest_face[2]), int(largest_face[3])
        img = frame[y:y+h, x:x+w]
        name = "Unknown"
        face_encoding = self.encode_face(img).astype(np.float64)

        largest_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pr_live_person = self.livePerson(gray, largest_rect)

        for face_load_encoding, face_load_guid in zip(known_face_encodings, known_face_guids):
            face_load_encoding = face_load_encoding.astype(np.float64)  # Приведение типов данных к float32
            if self._disType == 0: # COSINE
                cosine_score = self.face_recognizer.match(face_encoding, face_load_encoding, self._disType)
                min_distance = cosine_score
                if cosine_score < min_distance:
                    guid = face_load_guid
                    min_distance = cosine_score
                    #break
            else: # NORM_L2
                norml2_distance = self.face_recognizer.match(face_encoding, face_load_encoding, self._disType)
                if norml2_distance < min_distance:
                    guid = face_load_guid
                    min_distance = norml2_distance
                    name = "Successfully"
                    confidence = 1 - min_distance
                       
            if self._disType == 0: # COSINE
                if cosine_score < self._threshold_cosine:
                    name = "Unknown"
                    guid = None
                    confidence = 0.0

            else:
                if norml2_distance > self._threshold_norml2:
                    name =  "Unknown"
                    guid = None
                    confidence = 0.0


            end_time = time.time()  # End time for recognizing this frame
            recognition_time = end_time - start_time    
            print(f"Recognition time for current frame: {recognition_time:.2f} seconds")    
         
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)     

        return frame, guid, confidence, pr_live_person
    

    #================================================================================    
    def register_user(self, cap):
        countdown = 3  # Время отсчета в секундах
        start_time = time.time()
        text = ""
        guid = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)
 
            input_size = (frame.shape[1], frame.shape[0])
            self.detector.setInputSize(input_size)
            faces = self.detector.detect(frame)
            faces = faces[1]

            if remaining_time < 0:
                if not faces:
                    text = "No face detected. Please try again."
                    break

                largest_face = None
                max_area = 0
                for face in faces:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    area = w * h
                    if area > max_area:
                        max_area = area
                        largest_face = face

                if largest_face is None:
                    text = "No face detected. Please try again."
                    break

                x, y, w, h = int(largest_face[0]), int(largest_face[1]), int(largest_face[2]), int(largest_face[3])
                face_image = frame[y:y+h, x:x+w]
                result, guid  = self.add_face_to_db(face_image)
                end_time = time.time()
                registration_time = end_time - start_time
                text = "Registration successfully"
                print(f"Registration time: {registration_time:.2f} seconds")
                break
 
            largest_face = None
            max_area = 0
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face = face

            if largest_face is not None:    
                x, y, w, h = largest_face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Отображение времени отсчета на экране
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
 