# cs-fairness

## Setup
The file `download.py` contains the Google drive link to our trained generative models and test images. To automate
the setup process, from the main directory, run the following:
```
git submodule update --init --recursive
python3.6 -m venv env
source env/bin/activate
pip install -U pip
pip install -r requirements.txt
python download.py
tar -zxvf data.tar.gz
```

## Scripts
Sample scripts for running experiments can be found in `scripts/`
