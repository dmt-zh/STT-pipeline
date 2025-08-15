## ClearML Tracking Server

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml-server.jpg" width="850" height="500"/>


### Подготовка удаленного сервера

Подключаемся по ssh к удаленному серверу (например, в моем примере удаленный сервер ubuntu@89.169.180.52):
```sh
ssh ubuntu@89.169.180.52
```

Обновляем пакеты:
```sh
sudo apt update -y
```

Устанавливаем git:
```sh
sudo apt install git
```

Устанавливаем `docker` и `docker compose` по инструкциям из [документации](https://docs.docker.com/engine/install/ubuntu/#uninstall-old-versions):
```sh
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

```sh
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

```sh
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```sh
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```


### Конфигурация CLearML сервера

Копируем репозиторий:
```sh
git clone https://github.com/dmt-zh/STT-pipeline.git && cd STT-pipeline/server
```

Создаем `.env` файл на основании файла `.env.example`:
```sh
cp .env.example .env
```

С помощью любого текстового редактора открываем `.env` файл и каждой переменной присваиваем значение, например:
```yml
# Путь где будут расположены служебные директории и файлы
CLEARML_ROOT_DIR='/home/clearml/app'
# IP сервера для доступа через браузер
CLEARML_HOST_IP=89.169.180.52
# Имя для аутентификации пользователя
CLEARML_USERNAME=admin
# Пароль пользователя для аутентификации
CLEARML_PASSWORD=admin12345

###############################################################################

# Ключ доступа к S3
AWS_ACCESS_KEY_ID=awskey12345
# Секретный ключ S3
AWS_SECRET_ACCESS_KEY=awssecret12345
# Логин для аутентификации в MinIO
MINIO_USER=s3-admin
# Пароль пользователя для аутентификации MinIO
MINIO_PASSWORD=minioadmin1234
# Путь где будет расположено S3 хранилище
MINIO_ROOT_DIR='/home/clearml/app/s3-storage'
# IP сервера:порт для подключения S3 хранилища к ClearML
MINIO_HOST=89.169.180.52:9000

###############################################################################
# Переменные для обеспечения безопасности ClearML сервера
# Пример скрипта для генерации рандомных паролей:
# https://github.com/clearml/clearml-server/blob/master/apiserver/service_repo/auth/utils.py
CLEARML_API_ACCESS_KEY=coolapikey789
CLEARML_API_SECRET_KEY=coolsecretkey1236
CLEARML_AGENT_ACCESS_KEY=coolagentaccesskye12345
CLEARML_AGENT_SECRET_KEY=coolagentsecretkey12456
SECURE_HTTP_SESSION_SECRET=coolhttpsessionkey12345
SECURE_AUTH_TOKEN_SECRET=cooltokensecretkey12345
SECURE_APISERVER_USER_KEY=coolsecretuserkey12345
SECURE_APISERVER_USER_SECRET=coolsecretkeysuper9874
SECURE_WEBSERVER_USER_KEY=coolsecretkeyweb12345
SECURE_WEBSERVER_USER_SECRET=coolsupersecretkey12345
SECURE_TESTS_USER_KEY=coolsecretuserkeytest4569
SECURE_TESTS_USER_SECRET=coolsecretuserkey0147
```

Добавляем права запуска скрипту `setup.sh`
```sh
chmod +x setup.sh 
```

Запускаем скрипт подготовки окружения:
```sh
sudo ./setup.sh
```
* `setup.sh` запускается только при первом запуске `clearml-server`

### Запуск ClearML сервера

После завершения работы скрипта `setup.sh`, запускаем сервер ClearML:
```
sudo ./run.sh
```

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/server_start.jpg" width="850" height="500"/>


Для того чтобы убедится, что сервер успешно запустился смотрим логи сервера в контейнер `clearml-apiserver`, в логах должно быть сообщение что сервер вернул статус код 200:
```sh
sudo docker logs clearml-apiserver --tail 50 -f
```

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/apiserver_200.jpg" width="850" height="500"/>


После того как сервер запустился, в строке браузера вводим IP адрес сервера с портом 8080

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/webserver_ui.jpg" width="850" height="500"/>

Произойдет перенаправление на страницу аутентификации. Необходимо войти в сервис под пользователем и паролем, которые указаны в `.env` файле:
```
Username: admin (CLEARML_USERNAME)
Password: admin12345 (CLEARML_PASSWORD)
```

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_auth.jpg" width="850" height="500"/>


После аутентификации попадаем на главную страницу
<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_dashboard.jpg" width="850" height="500"/>


Из главной страницы можно убрать дефолтные примеры в разделе `Settings → Configuration`

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_settings.jpg" width="850" height="500"/>


На главной странице выбрать раздел `My Work`

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_mywork.jpg" width="500" height="500"/>


### Остановка ClearML сервера

Для остановки сервера ClearML необходимо выполнить команду:
```
sudo ./stop.sh
```
