## Автоматизация ML экспериментов с помощью CLearML

ClearML — MLOps платформа с открытым исходным кодом, предназначенная для автоматизации, организации и управления всем жизненным циклом машинного обучения. Предоставляет инструменты для управления экспериментами, умеет логировать эксперименты, оптимизировать гиперпараметры и хранить код.

**Базовый пайплайн состоит из двух компонентов:**
1. ClearML Tracking Server (Server). [Деплой ClearML Server](https://github.com/dmt-zh/STT-pipeline/tree/main/server)
2. GPU сервер для дообучения Whisper (Development Machine). [Дообучение Whisper](https://github.com/dmt-zh/STT-pipeline/tree/main/whisper)

<img src="https://github.com/dmt-zh/STT-pipeline/blob/main/static/clearml_logging.jpg" width="850" height="500"/>


Документация по работе с [CLearML](https://clear.ml/docs/latest/docs/references/sdk/task/)