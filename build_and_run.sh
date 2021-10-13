#! /bin/bash

docker-compose build
STAT_BUILD=$?
if [ ! ${STAT_BUILD} -eq 0 ]; then
	echo "[ERROR] \"docker-compose build\" is failed: (exit-status = ${STAT_BUILD})"
	exit
fi

docker-compose up -d
STAT_UP=$?
if [ ! ${STAT_UP} -eq 0 ]; then
	echo "[ERROR] \"docker-compose up -d\" is failed: (exit-status = ${STAT_UP})"
	exit
fi

docker-compose exec web python manage.py makemigrations --no-input
STAT_MAKEMIGRATIONS=$?
if [ ! ${STAT_MAKEMIGRATIONS} -eq 0 ]; then
	echo "[ERROR] \"docker-compose exec web python manage.py makemigrations --no-input\" is failed: (exit-status = ${STAT_MAKEMIGRATIONS})"
	exit
fi

docker-compose exec web python manage.py migrate --no-input
STAT_MIGRATE=$?
if [ ! ${STAT_MIGRATE} -eq 0 ]; then
	echo "[ERROR] \"docker-compose exec web python manage.py migrate --no-input\" is failed: (exit-status = ${STAT_MIGRATE})"
	exit
fi

docker-compose exec web python manage.py collectstatic --no-input
STAT_COLLECTSTATIC=$?
if [ ! ${STAT_COLLECTSTATIC} -eq 0 ]; then
	echo "[ERROR] \"docker-compose exec web python manage.py collectstatic --no-input\" is failed: (exit-status = ${STAT_COLLECTSTATIC})"
	exit
fi

docker-compose ps

