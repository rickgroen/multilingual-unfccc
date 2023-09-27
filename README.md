# Progress on Climate Action: a Multilingual Machine Learning Analysis of the Global Stocktake

## Dependencies
A [requirements](requirements.txt) file is available to retrieve all dependencies. Create a new python environment and install using:
```shell
pip install -r requirements.txt
``` 
Then you need to install the punkt and stopwords nltk corpora from a terminal running python (in your conda environment).
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```


## Downloading our data
First, fill out which **DATA_FOLDER** you are using in [config.py](config.py).

Then, if you are using a Windows machine, navigate to [our Zenodo repository](https://zenodo.org/record/8379988) and download *embeddings.zip* and *overview.csv*. 
Put both these files in your **DATA_FOLDER**. Extract *embeddings.zip*.

If you are using Linux, our data can be downloading and extracted by running
```
sh ./utils/download.sh
```


## Running our code
Ensure **DATA_FOLDER** points to where you store your data, then run [main.py](main.py). 
