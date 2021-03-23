# permutations_language

EMNLP code to build a language model using :  https://github.com/google-research/google-research/tree/master/perturbations and huggingface


```
pip install tensorflow --upgrade
pip install tensorflow_probability
pip install transformers
pip install datasets
```


## Usage

`python run.py`

## Organization

The file `fast_soft_trainer.py` contains a huggingface Trainer with the FenchelYoungLoss.
