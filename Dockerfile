FROM continuumio/miniconda3:4.7.12 
RUN conda install -y h2o jupyter &&\ 
pip install h2o &&\ 
pip install deap update_checker tqdm stopit &&\
pip install tpot &&\
pip install setuptools &&\ 
pip install wheel &&\
pip install mlbox &&\
pip install seaborn &&\
apt-get update &&\
apt-get -f install -y build-essential swig &&\
conda install gxx_linux-64 gcc_linux-64 swig &&\
apt-get install -y curl wget &&\
conda install -c conda-forge 'swig<4' &&\
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install &&\
pip install 'pyrfr>=0.8' &&\
pip install auto-sklearn
RUN mkdir pyautoweka
COPY /pyautoweka /pyautoweka
cd pyautoweka &&\
python setup.py install &&\
