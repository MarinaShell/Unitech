## Название проекта
#### Создание прототипа системы регистрации и распознавания пользователей по лицу

### Описание проекта
#### Проект предназначен для регистрации пользователя в базу данных sqllite, распознавания человека, определения его "живости". Проект написан на языке Python. Выбрана модель face_recognition из dlib. Для регистрациии и распознавания пользователя берется самое большое лицо в кадре. При распознавании пользователя сразу определяется "живость" лица в течение некоторого промежутка времени, которе можно задавать. При регистрации пользователя необходимо ввести его имя, при удачной операции - администратору возвращается guid пользователя, сгенерированный автоматически. При удачном распознавании пользователя администратору возвращается guid и имя пользователя.  

### Структура проекта

#### _cls_base_class.py_
##### Абстракный класс, в котором обозначены все методы, необходимые для работы. В данном классе определена общая логика работы с распознаванием человека, созданием базы данных в sqllite, удаление пользователей из базы данных, просмотр всех пользователей, выборка пользователя из базы данных по id, вставка пользователя в базу данных.

#### _cls_facerecognition.py_
##### Класс-наследник для абстракного класса, в котором определены свои конкретные функции для детекции и распознавания, занесения и извлечения embeddings в базе данных. В данном случае используется модель face_recognition из dlib для детекции и распознавания лица.

#### _cls_interface_work.py_
##### Класс интерфейса для организации процесса регистрации пользователя, распознавания, просмотра всех пользователей в базе данных, удаления всех пользователей в базе данных, удаления пользователя по id, установка времени детекции и включения определения живости лица.

#### _cls_life_person.py_
##### Класс определения живости лица. Включает 5 методов: моргание глаз, движение глаз, микродвижения головы, тексутра кожи, движение лицевой части в области ушей, глаз, носа.

#### _main.py_
##### Запуск объекта класса интерфейса.

### Для работы системы необходимо скачать файл python-3.12.5-amd64.exe с официального сайта https://www.python.org/downloads/ и установить в нужную папку. Кроме этого, в эту же папку необходимо установить все библиотеки из файла requirements.txt
