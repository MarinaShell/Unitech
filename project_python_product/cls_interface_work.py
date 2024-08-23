import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from cls_deepface import DeepFaceRecognition
from cls_lbph_opencv import LBPHFaceRecognition
from cls_facerecognition import FaceRecognition
from cls_dnn_opencv import DNNRecognition
from cls_insightface import InsightFaceRecognition
from cls_mtcnn import MTCNNFaceRecognition

import threading
from tkinter import simpledialog, messagebox, filedialog

class FaceRecognitionApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.recognition_thread = None
        self.stop_event = threading.Event()  # Используем Event для остановки потока

        self.app_face = FaceRecognition(db_path='face_rec.db')
        self.configure(bg="lightblue")  # Установка фона для окна
        self.geometry("180x750")  # Устанавливаем размер окна 800x600 пикселей
 
        self.title("Face Recognition App")

        tk.Button(self, text="Register User", command=self.register_user).pack(pady=10)
        tk.Button(self, text="Recognize Users", command=self.recognize_users).pack(pady=10)
        tk.Button(self, text="Show All Users", command=self.show_all_users).pack(pady=10)
        tk.Button(self, text="Delete User by ID", command=self.delete_user).pack(pady=10)
        tk.Button(self, text="Delete All Users", command=self.delete_all_users).pack(pady=10)
        tk.Button(self, text="Set Time for ActivateAlive", command=self.activateAlive).pack(pady=10)
        tk.Button(self, text="Stop ActivateAlive", command=self.stopActivateAlive).pack(pady=10)
        tk.Button(self, text="Exit", command=self.quit).pack(pady=10)

        self.recognition_thread = None
    #================================================================================
    def get_recognition_instance(self):
        return self.app_face
  
    #================================================================================
    def register_user(self):
        app = self.get_recognition_instance()
        guid = app.register_users_from_camera()
        if guid:
            messagebox.showinfo("Info", f"User {guid} registered successfully.")
        else:
            messagebox.showinfo("Info", f"User not registered.")

 
    #================================================================================
    def recognize_users(self):
        # Выбираем метод распознавания
        app = self.get_recognition_instance()       
        guid, confidence = app.recognize_users_from_camera()
        if guid:
            messagebox.showinfo("Info", f"User {guid} recognized successfully with {confidence*100}%.")
        else:
            messagebox.showinfo("Info", f"User not recognized.")
    
    #================================================================================
    def show_all_users(self):
        app = self.get_recognition_instance()
        users = app.list_users()
        user_list = "\n".join([f"ID: {user[0]}, Guid: {user[1]}" for user in users])
        messagebox.showinfo("All Users", user_list)

    #================================================================================
    def delete_user(self):
        app = self.get_recognition_instance()
        user_id = simpledialog.askinteger("Input", "Enter the user ID to delete:")
        if user_id:
            result = app.delete_user_by_id(user_id)
            messagebox.showinfo("Info", result)
    
    #================================================================================
    def delete_all_users(self):
        app = self.get_recognition_instance()
        result = app.deleteAllFromData()
        messagebox.showinfo("Info", result)        

    #================================================================================
    def activateAlive(self):
        app = self.get_recognition_instance()
        time = simpledialog.askinteger("Input", "Enter the time in sec for definition live people:")
        if time > 0:
            result = app.setAlivePerson(time)
            messagebox.showinfo("Info", result)

    #================================================================================
    def stopActivateAlive(self):
        app = self.get_recognition_instance()
        app.setAlivePerson(0)
