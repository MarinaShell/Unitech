import cv2
import sqlite3
import time
from abc import ABC, abstractmethod
import threading
import queue

class FaceRecognitionBase(ABC):
    def __init__(self, db_path):
        self.db_path = db_path
        self.create_db()
        self.stop_recognition = False  # ��������� ��� ���������� ��������� �������������
        self.current_frame = None
        self.video_path = ""
        self.isAlivePerson = True
        self.timeAlivePerson = 25

    #================================================================================
    def create_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      guid TEXT UNIQUE, 
                      name,
                      encoding BLOB)''')
        conn.commit()
        conn.close()
    
    #================================================================================
    @abstractmethod
    def load_known_faces(self):
        pass

    #================================================================================
    def delete_user_by_id(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        return f"User with id {user_id} deleted successfully."
   
    #================================================================================
    def delete_user_by_guid(self, user_guid):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE guid=?", (user_guid,))
        conn.commit()
        conn.close()
        return f"User with id {user_guid} deleted successfully."
        
    #================================================================================
    def list_users(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, guid, name FROM faces")
        rows = c.fetchall()
        conn.close()
        return rows
    
    #================================================================================
    @abstractmethod
    def encode_face(self, face_image):
        pass
    
    #================================================================================
    @abstractmethod
    def add_face_to_db(self, face_image):
        pass
    
    #================================================================================
    @abstractmethod
    def recognize_face(self, frame):
        pass

    #================================================================================
    def recognize_users_from_camera(self):
        self.stop_recognition = False
        cap = cv2.VideoCapture(0)
        return self.recognize_users(cap)

    #================================================================================
    def recognize_users_from_video(self):
        self.stop_recognition = False
        cap = cv2.VideoCapture(self.video_path)
        return self.recognize_users(cap)
    
    #================================================================================  
    def register_users_from_camera(self, name):
        self.stop_recognition = False
        cap = cv2.VideoCapture(0)
        return self.register_user(name, cap)

    #================================================================================
    def register_users_from_video(self, name):
        self.stop_recognition = False
        cap = cv2.VideoCapture(self.video_path)
        return self.register_user(name, cap)
    
    #================================================================================    
    @abstractmethod
    def register_user(self):
        pass

    #================================================================================
    def stop_recognition_process(self):
        self.stop_recognition = True
    
    #================================================================================    
    def recognize_users(self, cap):
        guid = 0
        confidence = 0.0
         
        start_time_all = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            start_time = time.time()
            recognized_frame, guid, confidence, islivePerson, name = self.recognize_face(frame_rgb)
            end_time = time.time()  # End time for recognizing this frame
            recognition_time = end_time - start_time
            #print(f"Recognition time for current frame: {recognition_time:.2f} seconds")          
          
            # Display the frame
            frame_rgb = cv2.cvtColor(recognized_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            self.current_frame = frame_rgb
            cv2.imshow('Recognize Users', frame_rgb)   
            
            if guid and islivePerson:
                    break
            else:
                current_time = time.time()
                elapsed_time = current_time - start_time_all
                if elapsed_time >= self.timeAlivePerson:
                    break
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return guid,confidence,name

    #================================================================================    
    def get_current_frame(self):
        return self.current_frame

    #================================================================================
    def exportFromData(self, guid):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT guid, name, encoding FROM faces WHERE guid=?", (guid,))
        user = c.fetchone()
        conn.close()
        return user
    
    #================================================================================
    def importToData(self, guid, encoding):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO faces (guid,name, encoding) VALUES (?, ?, ?)", (guid, name, encoding))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return f"User added successfully with id {user_id}."

    #================================================================================
    def getVideo(self):
        while not self.stop_recognition:
            frame = self.get_current_frame()
            if frame is not None:
                return frame
            time.sleep(0.03)  # ������� ��������� ����� ��������� ��������

        cv2.destroyAllWindows()
 
    #================================================================================        
    def setAdressCamera(self, video_path):
        self.video_path = video_path
    
    #================================================================================    
    def beginScanThread(self, result_queue):
        self.stop_recognition = False
        cap = cv2.VideoCapture(self.video_path)
        result = self.recognize_users( cap)
        result_queue.put(result)
        cap.release()
     
    #================================================================================    
    def beginScan(self):
        result_queue = queue.Queue()
        recognition_thread = threading.Thread(target=self.beginScanThread, args=("", result_queue))
        recognition_thread.start()
        recognition_thread.join()  # ���� ���������� ������
        return result_queue.get()  # �������� ��������� �� �������
    
    #================================================================================    
    def faceRecognitionAllThread(self, result_queue):
        self.stop_recognition = False
        cap = cv2.VideoCapture(self.video_path)
        result = self.register_user(cap)
        result_queue.put(result)
        cap.release()
     
    #================================================================================    
    def faceRecognitionAll(self):
        result_queue = queue.Queue()
        recognition_thread = threading.Thread(target=self.faceRecognitionAllThread, args=(result_queue))
        recognition_thread.start()
        recognition_thread.join()  # ���� ���������� ������
        return result_queue.get()  # �������� ��������� �� �������
    
    #================================================================================
    def deleteDataByGuid(self, guid):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE guid=?", (guid,))
        conn.commit()
        conn.close()
        return f"User with id {guid} deleted successfully."

    #================================================================================
    def deleteDataById(self, id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE guid=?", (id,))
        conn.commit()
        conn.close()
        return f"User with id {guid} deleted successfully."
    
    #================================================================================
    def deleteAllFromData(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM faces")
        c.execute("DELETE FROM sqlite_sequence WHERE name='faces'")  # ����� ��������������
        conn.commit()
        conn.close()
        return "All users deleted successfully."
    
       
    #================================================================================
    def setAlivePerson(self, time_sec):
        self.isAlivePerson = True
        self.timeAlivePerson = time_sec
        if time_sec == 0:
             self.isAlivePerson = False