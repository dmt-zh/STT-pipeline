Устанавливаем пакетный менеджер [uv](https://docs.astral.sh/uv/getting-started/installation/):
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Устанавливаем ruff:
```sh
uv tool install ruff@latest
```

Проверка синтаксиса:
```sh
ruff check
```

Безопасное исправление ошибок:
```sh
ruff check --fix
```

Принудительное исправление ошибок:
```sh
ruff check --fix --unsafe-fixes
```

Для корректной работы инференса для Whisper, необходимо добавить корректные пути к файлам cuDNN
```sh
echo export LD_LIBRARY_PATH=`uv run python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'` >> ~/.bashrc
```

Чтобы изменения внесённые в файл `bashrc` в текущем сеансе терминала вступили в силу, файл необходимо перезагрузить:
```sh
source ~/.bashrc
```
