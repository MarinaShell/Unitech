#### В папке «загрузчик Python 3.12.5» находится ссылка на установочный файл python-3.12.5-amd64.exe.

#### В папке «код с#» находится Program.cs -тестовый код для работы приложения. В нем необходимо указать правильные пути к директории с установленными библиотеками и папкой files.
##### _Для регистрации и распознавания пользователя необходимо указать адрес камеры (frs.setAdressCamera(source)). Для распознавания пользователя можно варьировать показателем «живости лица» (frs.setAlivePerson(25)), указав время в секундах. Если указать 0, то определение живости лица снимается. Регистрация пользователя и его распознавание происходит автоматически при обнаружении лица и выборе самого большого лица в камере. При регистрации пользователя необходимо ввести имя. При успешной регистрации функция возвращает guid этого пользователя, сформированный автоматически. При распознавании лица функция возвращает имя, guid пользователя и confidence._ 

#### В папке «установка Python кода» находится папка files, в которой находятся файлы Python кода и ссылка на файл для обнаружения «живости» лица. Эту папку files необходимо скопировать в директорию, где будут установлены Python и библиотеки. 

#### В папке «установка зависимостей» находится requirements.txt - список всех библиотек, необходимых для работы системы.

