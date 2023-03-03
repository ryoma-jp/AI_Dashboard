# --- Builder stage ---
FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3 as builder

RUN mkdir -p /home/app
RUN groupadd app ; useradd app -g app
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update ; \
      apt install -y postgresql gcc python3-dev musl-dev graphviz

RUN pip install --upgrade pip
RUN pip install flake8

# --- copy data without media ---
COPY ./app ./app
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
RUN groupadd app ; useradd app -g app

ENV HOME=/home/app
ENV APP_HOME=/home/app/web
ENV ENV_HOME=/home/app/env
RUN mkdir -p ${APP_HOME}
RUN mkdir -p ${APP_HOME}/static
RUN mkdir -p ${ENV_HOME}
WORKDIR ${APP_HOME}

RUN apt update ; \
    apt -y install libpq-dev netcat graphviz libgl1-mesa-dev psmisc usbutils
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/*

# --- copy data without media ---
COPY ./app $APP_HOME/app
COPY ./entrypoint.sh $APP_HOME/entrypoint.sh
COPY ./manage.py $APP_HOME/manage.py
COPY ./project $APP_HOME/project
COPY ./env/web/requirements.txt $APP_HOME/requirements.txt
COPY ./static $APP_HOME/static
COPY ./templates $APP_HOME/templates

RUN chown -R app:app ${APP_HOME}
RUN chown -R app:app ${ENV_HOME}

USER app

ENTRYPOINT ["/home/app/web/entrypoint.sh"]
