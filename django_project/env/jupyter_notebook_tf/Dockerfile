
# --- build command ---
#   docker build --network=host -f Dockerfile -t jupyter_nb/tensorflow_gpu .

# --- for machine learning ---
FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

RUN apt update ; \
    apt install -y libgl1-mesa-dev fonts-ipaexfont vim graphviz

RUN pip install setuptools==57.4.0
RUN pip install opencv-python==4.6.0.66 \
                natsort==8.2.0 numpy==1.23.5 pandas==1.5.2 \
                xfeat==0.1.1 seaborn==0.12.1 japanize-matplotlib==1.1.3 umap-learn==0.5.3 \
                optuna==3.0.4 scikit-learn==1.1.3 \
                sphinx==5.3.0 sphinx_rtd_theme==1.1.1 \
                mne==1.2.3 plotly==5.11.0 \
                graphviz==0.20.1

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -U requests==2.28.1

RUN pip install -U datasets==2.7.1 transformers[ja]==4.25.1
RUN pip install -U ginza==5.1.2 ja_ginza==5.1.2

COPY ./jupyter_notebook_config.py /root/.jupyter/

