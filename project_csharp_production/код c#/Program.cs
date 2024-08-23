// Version 1.1 - Initial version (2024.07.07)
// Version 1.3 - Fixed (2024.07.14)

using Python.Runtime;
using System;
using System.Linq;
using System.Text;
using System.Threading;
using System.Xml.Linq;

class Program
{
    static bool exitFlag = false;

    static void Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        // Установка переменных среды Python
        SetPythonEnvironment();

        // Путь к директории Scripts (Windows) виртуального окружения, где находится python312.dll
        string pythonDllPath = @"d:\marina\env_face\python312.dll"; // Укажите правильный путь
        pythonDllPath = Environment.ExpandEnvironmentVariables(pythonDllPath);

        // Укажите путь к python39.dll
        Runtime.PythonDLL = pythonDllPath;

        // Инициализация Python runtime
        PythonEngine.Initialize();


        // Путь к Python скрипту с классом FaceRecognitionSystem
        // Находится по пути: %USERPROFILE%\AppData\Local\Programs\Python\Python39\
        string scriptPath = @"d:\marina\env_face\files";

 
        // Загрузка и выполнение Python скрипта
        try
        {
            using (Py.GIL())
            {
                // Добавляем путь к директории с Python файлами в sys.path
                dynamic sys = Py.Import("sys");
                sys.path.append(scriptPath);

                // Импортируем необходимые модули
                dynamic faceRecognitionModule = Py.Import("cls_facerecognition");

                // Создаем экземпляр класса FaceRecognition
                dynamic faceRecognition = faceRecognitionModule.FaceRecognition(@"d:\marina\env_face\files\face.db", @"d:\marina\env_face\files");

                while (!exitFlag)
                {
                    Console.WriteLine("Выберите режим работы:");
                    Console.WriteLine("[1] Распознавание");
                    Console.WriteLine("[2] Регистрация пользователя");
                    Console.WriteLine("[3] Просмотр всех пользователей в БД");
                    Console.WriteLine("[4] Удаление по id");
                    Console.WriteLine("[5] Удаление всех пользователей из БД");
                    Console.WriteLine("Нажмите 'x' для выхода.");

                    string input = Console.ReadLine();

                    if (input.ToLower() == "x")
                    {
                        exitFlag = true;
                    }
                    else if (input == "1")
                    {
                        Console.WriteLine("РЕЖИМ РАСПОЗНАВАНИЯ\n");
                        RunRecognitionMode(faceRecognition);
                    }
                    else if (input == "2")
                    {
                        Console.WriteLine("РЕЖИМ РЕГИСТРАЦИИ\n");
                        Console.WriteLine("Введите имя пользователя: ");
                        string name = Console.ReadLine();
                        RunDataCollectionMode(name, faceRecognition);
                    }
                    else if (input == "3")
                    {
                        Console.WriteLine("РЕЖИМ ПРОСМОТРА ВСЕХ ПОЛЬЗОВАТЕЛЕЙ В БД\n");
                        ViewAllUserInBd(faceRecognition);
                    }
                    else if (input == "4")
                    {
                        Console.WriteLine("РЕЖИМ УДАЛЕНИЯ ПОЛЬЗОВАТЕЛЯ ПО ID\n");
                        Console.WriteLine("Введите ID пользователя: ");
                        string id = Console.ReadLine();
                        DeleteUserByNameMode(faceRecognition, id);
                    }
                    else if (input == "5")
                    {
                        Console.WriteLine("РЕЖИМ УДАЛЕНИЯ ВСЕХ ПОЛЬЗОВАТЕЛЕЙ ИЗ БД\n");
                        DeleteAllUsersMode(faceRecognition);
                        Console.WriteLine("Все пользователи удалены ");
                    }
                    else
                    {
                        Console.WriteLine("Неверный выбор. Попробуйте снова.");
                    }

                    // Пауза между режимами. Имитация перерыва между процедурами.
                    if (!exitFlag)
                    {
                        Console.WriteLine("Пауза между режимами. Ждем 3 секунды...");
                        Thread.Sleep(3000);
                        Console.WriteLine("Пауза завершена.");
                    }
                }
            }
        }
        catch (PythonException ex)
        {
            Console.WriteLine("Ошибка выполнения Python скрипта: " + ex.Message);
        }

        // Деинициализация движка Python
        PythonEngine.Shutdown();

        // Пауза перед завершением программы
        Console.WriteLine("\nНажмите любую клавишу для выхода...");
        Console.ReadKey();
    }

    /***************************************************************************/
    static void SetPythonEnvironment()
    {
        // Путь к директории, где находится python39.dll
        string pythonDllPath = @"%USERPROFILE%\AppData\Local\Programs\Python\Python312";
        pythonDllPath = Environment.ExpandEnvironmentVariables(pythonDllPath);

        // Получение текущего значения переменной PATH
        string currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";

        // Проверка, содержит ли текущий PATH путь к python39.dll
        if (!currentPath.Split(';').Contains(pythonDllPath))
        {
            // Добавление пути к python39.dll в PATH
            Environment.SetEnvironmentVariable("PATH", currentPath + ";" + pythonDllPath);
        }
    }

    /***************************************************************************/
    // Режим "Распознования" для пользователя
    static void RunRecognitionMode(dynamic frs)
    {
        // Индекс к камере
        dynamic source = 0;
        frs.setAlivePerson(25);
        frs.setAdressCamera(source);

        // Вызов Python метода, который возвращает два значения
        dynamic result = frs.recognize_users_from_video();

        // Распаковка значений в C#
        dynamic guid = result[0];
        dynamic confidence = result[1];
        dynamic name = result[2];

        if (guid != 0)
            Console.WriteLine($"\n(C#) Пользователь {name} раcпознан с вероятностью {confidence*100}%");
        else
            Console.WriteLine($"\n(C#) Пользователь не распознан!");
    }

    /***************************************************************************/
    //Режим "Сбор данных" для пользователя
    static void RunDataCollectionMode(string name, dynamic frs)
    {
        // Индекс к камере
        dynamic source = 0;
        frs.setAdressCamera(source);

        // Вызов Python метода, который возвращает два значения
        dynamic guid = frs.register_users_from_video(name);

        if (guid!=0)
            Console.WriteLine($"\n(C#) Пользователю: {name} присвоен guid: {guid}");
        else
            Console.WriteLine($"\n(C#) Пользователь не зарегистрирован!");
    }

    /***************************************************************************/
    static void DeleteUserByNameMode(dynamic frs, string id)
    {
        // Удаление пользователя по имени из БД
        frs.deleteDataById(id);
    }

    /***************************************************************************/
    static void DeleteAllUsersMode(dynamic frs)
    {
        // Удаление всех пользователей из БД
        frs.deleteAllFromData();
    }

    /***************************************************************************/
    static void ViewAllUserInBd(dynamic frs)
    {
        // Вызов метода list_users
        dynamic users = frs.list_users();

        // Обработка результатов
        var userList = new List<Tuple<int, string, string>>();
        // Обработка результатов
        if (users != null)
        {

            foreach (var user in users)
            {
                int id = (int)user[0];
                string guid = (string)user[1];
                string name = (string)user[2];
                userList.Add(new Tuple<int, string, string>(id, guid, name));
            }
            if (userList.Count == 0)
            {
                // Вывод сообщения о том, что пользователей нет
                Console.WriteLine("Пользователи не найдены.");
            }
            else
            {
                // Вывод пользователей
                foreach (var user in userList)
                {
                    Console.WriteLine($"ID: {user.Item1}, GUID: {user.Item2}, NAME: {user.Item3}");
                }
            }
  
        }
    }

}

