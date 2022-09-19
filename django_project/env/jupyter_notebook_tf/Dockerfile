
# --- build command ---
#   docker build --network=host -f Dockerfile -t jupyter_nb/tensorflow_gpu .

# --- for machine learning ---
FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt update ; \
    apt install -y libgl1-mesa-dev fonts-ipaexfont vim

RUN pip install setuptools==57.4.0
RUN pip install opencv-python natsort numpy pandas xfeat seaborn japanize-matplotlib \
                optuna scikit-learn umap-learn \
                sphinx sphinx_rtd_theme


