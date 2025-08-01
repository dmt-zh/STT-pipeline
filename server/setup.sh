#!/bin/bash

echo 'Setting up environment for starting ClearML server.'

###############################################################################

# Формирование полного пути к текущей директории
BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}")"

###############################################################################

# Загрузка переменных окружения
set -o allexport
source "${BASE_DIR}/.env"
set +o allexport

###############################################################################

# Формирование пути к корневой директории системных файлов ClearML
ROOT_DIR="${CLEARML_ROOT_DIR:-/opt}"
echo -e "📁 System data of ClearML server will be stored in \"${ROOT_DIR}\" directory."

###############################################################################

# Изменение значения для Elasticsearch в Docker
# https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/#deploying

echo "vm.max_map_count=262144" > /tmp/99-clearml.conf
sudo mv /tmp/99-clearml.conf /etc/sysctl.d/99-clearml.conf
sudo sysctl -w vm.max_map_count=262144
sudo service docker restart

###############################################################################

# Формирование системных директорий ClearML и добавление прав доступа
CLEARML_ROOT_DIR="${ROOT_DIR}/clearml"
CLEARML_DATA_DIR="${CLEARML_ROOT_DIR}/data"
CLEARML_CONFIG_DIR="${CLEARML_ROOT_DIR}/config"

if [ -d "${CLEARML_ROOT_DIR}" ]; then
    echo -e "🚨 \"${CLEARML_ROOT_DIR}\" directory already exists."
    echo -e "\033[0;32m✔\033[0m Removing existing \"${CLEARML_ROOT_DIR}\" directory."
    sudo rm -R "${CLEARML_ROOT_DIR}/"
fi

sudo mkdir -p "${CLEARML_ROOT_DIR}/logs"
sudo mkdir -p "${CLEARML_CONFIG_DIR}"
sudo mkdir -p "${CLEARML_DATA_DIR}/elastic_7"
sudo mkdir -p "${CLEARML_DATA_DIR}/mongo_4/db"
sudo mkdir -p "${CLEARML_DATA_DIR}/mongo_4/configdb"
sudo mkdir -p "${CLEARML_DATA_DIR}/redis"
sudo mkdir -p "${CLEARML_DATA_DIR}/fileserver"

sudo chown -R 1000:1000 "${CLEARML_ROOT_DIR}"

###############################################################################

# Добавление прав на исполнение скриптов
sudo chmod +x "${BASE_DIR}"/run.sh
sudo chmod +x "${BASE_DIR}"/stop.sh

###############################################################################

# Обновляем значения key и secret_key для S3 хранилища в файле clearml.conf
BASE_CLEARML_CONFIG="${BASE_DIR}/configs/clearml.conf"
cp "${BASE_CLEARML_CONFIG}" "${CLEARML_CONFIG_DIR}"

sed -i -E "s/(key:)[[:space:]]*.*/\1 \"${AWS_ACCESS_KEY_ID}\"/" "${CLEARML_CONFIG_DIR}/clearml.conf"
sed -i -E "s/(secret:)[[:space:]]*.*/\1 \"${AWS_SECRET_ACCESS_KEY}\"/" "${CLEARML_CONFIG_DIR}/clearml.conf"
sed -i -E "s/(host:)[[:space:]]*.*/\1 \"${MINIO_HOST}\"/" "${CLEARML_CONFIG_DIR}/clearml.conf"
echo -e "\033[0;32m✔\033[0m Updated "clearml.conf" in \"${CLEARML_CONFIG_DIR}\" directory."

###############################################################################

# Устанавливаем значения в apiserver.conf для аутентификации
BASE_APISERVER_CONFIG="${BASE_DIR}/configs/apiserver.conf"
cp "${BASE_APISERVER_CONFIG}" "${CLEARML_CONFIG_DIR}"

sed -i -E "s/(username:)[[:space:]]*.*/\1 \"${CLEARML_USERNAME}\"/" "${CLEARML_CONFIG_DIR}/apiserver.conf"
sed -i -E "s/(password:)[[:space:]]*.*/\1 \"${CLEARML_PASSWORD}\"/" "${CLEARML_CONFIG_DIR}/apiserver.conf"
echo -e "\033[0;32m✔\033[0m Updated "apiserver.conf" in \"${CLEARML_CONFIG_DIR}\" directory."
