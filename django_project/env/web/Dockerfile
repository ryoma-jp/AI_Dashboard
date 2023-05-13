# --- Builder stage ---
FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3 as builder

RUN mkdir -p /home/app
RUN groupadd app && useradd app -g app
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update && \
      apt install -y postgresql gcc python3-dev musl-dev graphviz

RUN pip install --upgrade pip
RUN pip install flake8

# --- copy data without media ---
#COPY ./app ./app
COPY ./entrypoint.sh ./entrypoint.sh
COPY ./manage.py ./manage.py
COPY ./project ./project
COPY ./env/web/requirements.txt ./requirements.txt
COPY ./static ./static
COPY ./templates ./templates

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt


# --- Main stage ---
FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

RUN mkdir -p /home/app
RUN groupadd app && useradd app -g app

ENV HOME=/home/app
ENV APP_HOME=/home/app/web
ENV ENV_HOME=/home/app/env
RUN mkdir -p ${APP_HOME}
RUN mkdir -p ${APP_HOME}/static
RUN mkdir -p ${ENV_HOME}
WORKDIR ${APP_HOME}

RUN apt update && \
    apt -y install libpq-dev netcat graphviz libgl1-mesa-dev psmisc usbutils v4l-utils
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/*

# --- copy data without media (files) ---
COPY ./entrypoint.sh $APP_HOME/entrypoint.sh
COPY ./manage.py $APP_HOME/manage.py
COPY ./env/web/requirements.txt $APP_HOME/requirements.txt
COPY ./static $APP_HOME/static

# --- copy data without media (directories) ---
#  * if DEBUG, directories are NOT copied(bind in ``docker-compose.yml``).
#COPY ./app $APP_HOME/app
#COPY ./project $APP_HOME/project
#COPY ./templates $APP_HOME/templates

RUN chown -R app:app ${HOME}
RUN chown -R app:app ${APP_HOME}
RUN chown -R app:app ${ENV_HOME}

# --- get external tensorflow models ---
#ENV EXT_MODEL_PATH=${APP_HOME}/app/machine_learning/lib/trainer/tf_models/yolov3
ENV EXT_MODEL_PATH=/tmp

ENV MODEL_FILE=https://raw.githubusercontent.com/ryoma-jp/yolov3-tf2/b947b6d2dc227b9df8d5d38e6002d7dcc0124bb0/yolov3_tf2/models.py
RUN wget -P ${EXT_MODEL_PATH} ${MODEL_FILE}

ENV MODEL_FILE=https://raw.githubusercontent.com/ryoma-jp/yolov3-tf2/b947b6d2dc227b9df8d5d38e6002d7dcc0124bb0/yolov3_tf2/utils.py
RUN wget -P ${EXT_MODEL_PATH} ${MODEL_FILE}

USER app

ENTRYPOINT ["/home/app/web/entrypoint.sh"]
