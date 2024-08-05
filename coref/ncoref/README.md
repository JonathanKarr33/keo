### nueralcoref Setup

Github: https://github.com/huggingface/neuralcoref

1. Create conda environment with Python Version 3.6. Note it does not work with newer Python versions.
    1. `conda create --name ncoref python=3.6`
    2. `conda activate ncoref`
2. Install spaCy
    1. `pip install spacy==2.1.0 ` (doesn't work with newer versions)
    2. `python -m spacy download en`
3. Install nueralCoref
    1. `pip install neuralcoref` 
4. Download spaCy Model
    1. `python -m spacy download en_core_web_lg`
    2. `python -m spacy download en_core_web_sm` 
5. Install dependencies to run script
    1. `pip install pandas`
    2. `pip install textblob`
6. Run the script
    1. `python coref/ncoref/coref_faa.py` 

#### Reproducibility Rating:

<img src="../../star_clip.jpg" alt="Star" width="50" height="50"><img src="../../star_clip.jpg" alt="Star" width="50" height="50"><img src="../../star_clip.jpg" alt="Star" width="50" height="50"> <img src="../../star_clip.jpg" alt="Star" width="50" height="50">

*neuralcoref is deterministic*

Nueralcoref was easy to implement. All you need is a Conda environment and install spaCy and nueralcoref via pip. However, it doesn't work with the latest Python and SpaCy versions. In order to successfully run neuralcoref, Python version 3.6 and spaCy version 2.1 is needed.