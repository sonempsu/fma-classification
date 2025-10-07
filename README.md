# FMA progject notes
## Local FMA data path: `C:\Users\user\Desktop\data`
## How to set up windows environment:
## Setup (windows)
## use pyenv to install python 3.13.1(tcl dependency issue with 3.13.0)
pyenv shell 3.13.1
python -m venv env
.\env\Scripts\Activate
## install dependencies
pip install -r requirements.txt
## Environment Setup
source env/bin/activate
## Testing
python run_test.py

