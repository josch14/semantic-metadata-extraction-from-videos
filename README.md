# Semantic Metadata Extraction from Generated Video Captions

<p align="center">
  <img src="./resources/method_overview_1.png" width="800">
</p>

<p align="center">
  <img src="./resources/method_overview_2.png" width="800">
</p>

## Installation
The following installation steps are required in order to perform entity, property & relation extraction from text
or generated captioned events.

```
# create conda environment
conda create -n Video2Metadata python=3.7
conda activate Video2Metadata

# install spaCy with NeuralCoref from source (see https://github.com/huggingface/neuralcoref/issues/310)
cd src
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
cd ../../

# download spaCy language model
python -m spacy download en_core_web_lg

# install WordNet
conda install -c anaconda nltk

# package for querying the YouTube API for the category of a video (not necessarily required)
conda install -c conda-forge google-api-python-client
```

<br>

If there occurs an **error** when installing spaCy with NeuralCoref, then the following installation may work.

```
# create conda environment (after removing the old one)
conda create -n Video2Metadata python=3.7
conda activate Video2Metadata

# install spaCy with NeuralCoref via pip (see https://github.com/huggingface/neuralcoref/issues/209)
# note that the spaCy version is 2.1.0, which means that spaCy will be much slower.
pip install spacy==2.1.0
pip install neuralcoref

# download spaCy language model
python -m spacy download en_core_web_lg

# install WordNet
conda install -c anaconda nltk

# package for querying the YouTube API for the category of a video (not necessarily required)
conda install -c conda-forge google-api-python-client
```

## Entity, Property & Relation Extraction from Text
The entity, property & relation extraction methods can be applied on custom text. For this, pass some text to 
`extract_from_text.py` to extract and print the extracted semantic metadata.
```
python extract_from_text.py --text "A man is standing in front of a fridge. He opens it and takes out a red glass."
```

## Entity, Property & Relation Extraction from Captioned Events
To apply the semantic metadata extraction methods on captioned events (including temporal information) instead of text, 
you may add an example consisting of sentences and temporal segments to the given list of examples in 
`extract_from_captioned_events.py` (already included are the examples of entity, property & relation extraction 
from captioned events as presented in the paper).
```
python extract_from_captioned_events.py
```
