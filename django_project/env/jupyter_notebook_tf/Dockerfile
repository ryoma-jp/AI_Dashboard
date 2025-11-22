
# --- build command ---
#   docker build --network=host -f Dockerfile -t jupyter_nb/tensorflow_gpu .

# --- for machine learning ---
FROM tensorflow/tensorflow:2.20.0-gpu-jupyter

RUN apt update ; \
    apt install -y libgl1-mesa-dev fonts-ipaexfont vim graphviz

RUN pip install setuptools==57.5.0
RUN pip install opencv-python==4.12.0.88 \
                natsort==8.2.0 numpy==2.2.6 pandas==2.3.3 \
                xfeat==0.1.1 seaborn==0.13.2 japanize-matplotlib==1.1.3 umap-learn==0.5.8 \
                optuna==4.5.0 scikit-learn==1.7.2 \
                sphinx==8.2.3 sphinx_rtd_theme==3.0.2 \
                mne==1.10.2 plotly==6.3.1 \
                graphviz==0.21

#RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

RUN pip install -U requests==2.32.5

RUN pip install -U datasets==4.3.0 transformers[ja]==4.57.1
RUN pip install -U ginza==5.2.0 ja_ginza==5.2.0

COPY ./jupyter_notebook_config.py /root/.jupyter/

