import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import local_binary_pattern
import os

class LivePersonDetection:
    def __init__(self, db_path):
        predictor_path = os.path.join(db_path, "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(predictor_path)
        self.EAR_THRESHOLD = 0.3
        self.EAR_CONSEC_FRAMES = 3
        self.counter = 0
        self.total_blinks = 0
        self.blink_sequence = []
        self.prev_frame = None
        self.is_eye_blink = True
        self.is_eye_move = True
        self.is_movement = False
        self.is_skin = False
        self.is_point = False
        self.MOVEMENT_THRESHOLD = 3.0  # Установите подходящее значение порога движения
        self.init = True
        self.base_list = []
        self.alive_list = []
        self.LEN_BASE_LIST = 50 # количество учитываемых кадров для обнаружения движения
        self.BASE_TRESHOLD = 1 # пороговое значение изменения ключевых расстояний для определения движения
        self.LEN_ALIVE_LIST = 5 # количество учитываемых последних значений alive
        self.ALIVE_TRESHOLD = 2 # пороговое значение для определения живого лица (ALIVE_TRESHOLD из LEN_ALIVE_LIST должны быть true)
    
        
    #================================================================================    
    #вычисление коэффициента аспекта глаза (Eye Aspect Ratio, EAR), 
    #который используется для определения состояния глаз 
    def __eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    #================================================================================
    #нормализация лица
    def __normalize_face(self, shape, size=256):
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
    #проверка на моргание глаз
    def __detect_blink(self, shape):
        normalized_shape = self.__normalize_face(shape)
        
        left_eye = normalized_shape[36:42]
        right_eye = normalized_shape[42:48]
        
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        leftEAR = self.__eye_aspect_ratio(left_eye)
        rightEAR = self.__eye_aspect_ratio(right_eye)

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
    #проверка на движение глаз
    def __detect_eye_movement(self, shape):
        shape = self.__normalize_face(shape)
    
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
    
        eye_centers = np.mean([left_center, right_center], axis=0)
    
        # Инициализация предыдущих центров, если они не установлены
        if not hasattr(self, 'prev_eye_centers'):
            self.prev_eye_centers = eye_centers
            return False

        # Вычисление расстояния перемещения
        movement_distance = np.linalg.norm(eye_centers - self.prev_eye_centers)
        self.prev_eye_centers = eye_centers

        # Проверка, если движение глаз превышает порог
        if movement_distance > self.MOVEMENT_THRESHOLD:
            return True
    
        return False
   
    #================================================================================
    #анализ тексутры кожи
    def __analyze_skin_texture(self, face_image):
        lbp = local_binary_pattern(face_image, P=8, R=1, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist

    #================================================================================
    #проверка на текстуру кожи
    def __is_real_skin_texture(self, face_image):
        if face_image is None or face_image.size == 0:
            return False
    
        hist = self.__analyze_skin_texture(face_image)
        return np.sum(hist) > 0.5  # Простой порог для демонстрации

    #================================================================================
    #проверка на микродвижения
    def __detect_micro_movements(self, gray_frame):
        if self.init:
            self.prev_frame = gray_frame
            self.init = False
        if self.prev_frame is None:
             return False

        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        movement = np.mean(mag)

        self.prev_frame = gray_frame

        return movement > 0.1  # Порог для обнаружения микродвижений

    #================================================================================
    #проверка на движения ушей и глаз
    def __detect_points(self, result):    
        # Конвертация изображения в RGB
        keypoints = result.location_data.relative_keypoints

        line_1 = ((keypoints[0].x - keypoints[4].x) ** 2 + (keypoints[0].y - keypoints[4].y) ** 2) ** 0.5
        line_2 = ((keypoints[1].x - keypoints[5].x) ** 2 + (keypoints[1].y - keypoints[5].y) ** 2) ** 0.5

        base = line_1 / line_2 if line_2 > 0 else 0
        if base > 0:
            self.base_list.append(base)
        elif self.base_list:
            self.base_list.append(self.base_list[-1])

        if len(self.base_list) > self.LEN_BASE_LIST:
            self.base_list.pop(0)
        if max(self.base_list) - min(self.base_list) > self.BASE_TRESHOLD:
            self.alive_list.append(True)
        else:
            self.alive_list.append(False)
        if len(self.alive_list) > self.LEN_ALIVE_LIST:
            self.alive_list.pop(0)

        if sum(self.alive_list) < self.ALIVE_TRESHOLD:
            alive_face = False
        else:
            alive_face = True

        return alive_face

    #================================================================================
    def __detect_points_face(self, largest_face, frame_rgb):
        # Получаем ключевые точки лица (landmarks) с использованием face_recognition
        landmarks = face_recognition.face_landmarks(frame_rgb, [largest_face])

        if landmarks:
            # Извлекаем нужные ключевые точки (например, глаза и уголки рта)
            left_eye = landmarks[0]['left_eye']
            right_eye = landmarks[0]['right_eye']
            nose_tip = landmarks[0]['nose_tip']

            # Вычисление расстояний между ключевыми точками
            line_1 = ((left_eye[0][0] - nose_tip[0][0]) ** 2 + (left_eye[0][1] - nose_tip[0][1]) ** 2) ** 0.5
            line_2 = ((right_eye[0][0] - nose_tip[0][0]) ** 2 + (right_eye[0][1] - nose_tip[0][1]) ** 2) ** 0.5

            base = line_1 / line_2 if line_2 > 0 else 0
            if base > 0:
                self.base_list.append(base)
            elif self.base_list:
                self.base_list.append(self.base_list[-1])

            if len(self.base_list) > self.LEN_BASE_LIST:
                self.base_list.pop(0)
            if max(self.base_list) - min(self.base_list) > self.BASE_TRESHOLD:
                self.alive_list.append(True)
            else:
                self.alive_list.append(False)
            if len(self.alive_list) > self.LEN_ALIVE_LIST:
                self.alive_list.pop(0)

            if sum(self.alive_list) < self.ALIVE_TRESHOLD:
                alive_face = False
            else:
                alive_face = True

            return alive_face

    #================================================================================
    #на вход функции - фрейм,в котором надо определить на живость лицо
    def is_real_person(self, frame, rect):
        shape = self.predictor(frame, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        face_image = frame[y:y+h, x:x+w]

        pr_eye_move = False
        if self.is_eye_move:
            pr_eye_move = self.__detect_eye_movement(shape)
  
        pr_eye_blink = False
        if self.is_eye_blink:
            pr_eye_blink = self.__detect_blink(shape)
     
        pr_skin = False    
        if self.is_skin:
            pr_skin = self.__is_real_skin_texture(face_image)
                
        pr_movement = False    
        if self.is_movement:
            pr_movement = self.__detect_micro_movements(frame)
 
        pr_point = False       
        if self.is_point:
            # Преобразование кадра в формат RGB, так как face_recognition работает с RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Обнаружение лиц в кадре
            face_locations = face_recognition.face_locations(frame_rgb)
            
            largest_face = None
            max_area = 0
            
            # Проходим по всем обнаруженным лицам
            for face_location in face_locations:
                top, right, bottom, left = face_location
                
                # Вычисление размеров ограничивающей рамки
                bbox_width = right - left
                bbox_height = bottom - top
                bbox_area = bbox_width * bbox_height

                # Поиск самого большого лица
                if bbox_area > max_area:
                    largest_face = face_location
                    max_area = bbox_area
            
            if largest_face:
                # Передача самой большой области лица в функцию для дальнейшей обработки
                pr_point = self.__detect_points_face(largest_face)
                return pr_point

        # Объединяем результаты проверок в один общий параметр
        all_checks_passed = True
        if self.is_eye_blink:
            all_checks_passed = all_checks_passed and pr_eye_blink
        if self.is_eye_move:
            all_checks_passed = all_checks_passed and pr_eye_move
        if self.is_skin:
            all_checks_passed = all_checks_passed and pr_skin
        if self.is_movement:
            all_checks_passed = all_checks_passed and pr_movement
        if self.is_point:
            all_checks_passed = all_checks_passed and pr_point
         
        return all_checks_passed
    
    #================================================================================
    #установка функций, по которым надо определять живость, по умолчанию все включено
    def set_function(self,  eye_blink = True, 
                            eye_move = True, 
                            movement = True, 
                            skin = True, 
                            point = False):
        self.is_eye_blink = eye_blink
        self.is_eye_move = eye_move
        self.is_movement = movement
        self.is_skin = skin
        self.is_point = point
              
 