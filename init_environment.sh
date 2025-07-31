#!/bin/bash

###############################################################################

# Загрузка переменных окружения
set -o allexport
source .env
set +o allexport

###############################################################################

# Формирование пути к корневой директории системных файлов ClearML
ROOT_DIR="${CLEARML_ROOT_DIR:-/opt}"
echo "System data of ClearML server will be stored in '${ROOT_DIR}' directory."

###############################################################################

# Изменение значения для Elasticsearch в Docker
# https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/#deploying

echo "vm.max_map_count=262144" > /tmp/99-clearml.conf
sudo mv /tmp/99-clearml.conf /etc/sysctl.d/99-clearml.conf
sudo sysctl -w vm.max_map_count=262144
sudo service docker restart

###############################################################################

# Формирование системных директорий ClearML и добавление прав доступа
if [ -d "$DIRECTORY" ]; then
    sudo rm -R "${ROOT_DIR}"/clearml
fi

sudo mkdir -p "${ROOT_DIR}"/clearml/data/elastic_7
sudo mkdir -p "${ROOT_DIR}"/clearml/data/mongo_4/db
sudo mkdir -p "${ROOT_DIR}"/clearml/data/mongo_4/configdb
sudo mkdir -p "${ROOT_DIR}"/clearml/data/redis
sudo mkdir -p "${ROOT_DIR}"/clearml/logs
sudo mkdir -p "${ROOT_DIR}"/clearml/config
sudo mkdir -p "${ROOT_DIR}"/clearml/data/fileserver
sudo chown -R 1000:1000 "${ROOT_DIR}"/clearml

###############################################################################
