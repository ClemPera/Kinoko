# Kinoko

## Setup
- pyenv virtualenv 3.12.9 kinoko
- pyenv local kinoko
- pip install -r requirements.txt

## Datasets
- [image dataset](https://www.kaggle.com/datasets/derekkunowilliams/mushrooms?select=mushroom_dataset)
- [table dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

## Modification
`image_dataset` folder has been modified:
- remove `conditionally edible`
- merge `deadly` inside `poisonous`
- remove `Armillaria Mellea` and `Suillus granulatus` from `poisonous` (duplicated from `edible`)