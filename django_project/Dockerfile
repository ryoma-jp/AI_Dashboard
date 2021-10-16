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
COPY . .

COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

RUN mkdir -p /home/app
RUN groupadd app ; useradd app -g app

ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir -p ${APP_HOME}
RUN mkdir -p ${APP_HOME}/static
WORKDIR ${APP_HOME}

RUN apt update ; apt -y install libpq-dev netcat graphviz
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/*

COPY ./entrypoint.sh $APP_HOME

COPY . $APP_HOME

RUN chown -R app:app ${APP_HOME}

USER app

ENTRYPOINT ["/home/app/web/entrypoint.sh"]
