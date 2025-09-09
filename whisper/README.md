## Конфигурация GPU сервера:

Обновляем систему, ставим необходимые системные пакеты и перезагружаем сервер:
```sh
apt -y update && apt -y upgrade
apt -y install build-essential nvtop vim htop dkms
reboot
```

Устанавливаем библиотеки для обработки аудио:
```sh
sudo apt-get install -y portaudio19-dev
sudo add-apt-repository -y ppa:savoury1/ffmpeg4  
sudo apt install -y ffmpeg  
ffmpeg -version
```

Скачиваем и устанавливаем CUDA и драйвер (если драйвер уже установлен, то при установке `cuda` убрать флаг `--driver`):
```sh
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run;
sh cuda_12.6.0_560.28.03_linux.run --silent --toolkit --driver
```

Проверить что установка CUDA прошла успешно с помощью команды:
```sh
nvidia-smi
```

Устанавливаем пакетный менеджер [uv](https://docs.astral.sh/uv/getting-started/installation/):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Чтобы изменения внесённые в файл `bashrc` в текущем сеансе терминала вступили в силу, файл необходимо перезагрузить:
```
source ~/.bashrc
```

Устанавливаем линтер ruff:
```
uv tool install ruff@latest
```


## Конфигурация ClearML и запуск тренировки:

Копируем репозиторий:
```sh
git clone https://github.com/dmt-zh/STT-pipeline.git && cd STT-pipeline/whisper
```

Инициализируем виртуальное окружение и устанавливаем необходимые библиотеки:
```sh
uv sync
```

Для корректной работы инференса для Whisper, необходимо добавить корректные пути к файлам cuDNN
```
echo export LD_LIBRARY_PATH=`uv run python3 -c 'import importlib.resources; print(str(importlib.resources.files("nvidia.cublas.lib")) + ":" + str(importlib.resources.files("nvidia.cudnn.lib")._paths[0]))'` >> ~/.bashrc
```

Чтобы изменения внесённые в файл `bashrc` в текущем сеансе терминала вступили в силу, файл необходимо перезагрузить:
```
source ~/.bashrc
```

Активируем виртуальное окружение:
```sh
source .venv/bin/activate
```

В активированном виртуальном окружении инициализируем `clearml` командой:
```sh
clearml-init
```

В веб интерфейсе ClearML Tracking Server создаем креды

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_new_creds.jpg" width="500" height="500"/>

Копируем креды, вставляем в консоль и нажимаем `Enter`

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_creads.jpg" width="500" height="500"/>

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_cli_creds.jpg" width="500" height="500"/>


Деактивируем виртуальное окружение:
```sh
deactivate
```

Запускаем тренировку в фоновом режиме:
```sh
nohup ./train.py &> log_train.log &
```

В процессе тренировки будут логироваться различные значения, и после завершения тренировки в S3 будет залогированы артефакты.

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_scalars.jpg" width="850" height="500"/>

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_debug.jpg" width="850" height="500"/>



## Действие перед подготовкой PR-а

Проверка синтаксиса:
```
ruff check
```

Безопасное исправление ошибок:
```
ruff check --fix
```

Принудительное исправление ошибок:
```
ruff check --fix --unsafe-fixes
```
