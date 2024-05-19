# Source code for the master's thesis by Alfreds Lapkovskis & Natalia Nefedova (2024)

## Advancements in Agriculture: Multimodal Deep Learning for Enhanced Plant Identification
#### Developing a Multi-organ Plant Classifier Using Multimodal Fusion Architecture Search

##### Please also check our sample iOS app that utilizes the proposed model: https://github.com/AlfredsLapkovskis/MultimodalPlantClassifier-iOS

# 1. Setup

We used Python 3.11.5.

## 1.1. Configure a Virtual Environment

Execute these commands from the project root directory:
```zsh
python -m venv env

# This is for macOS. Please google how to activate venv for your OS.
source env/bin/activate

pip install -r requirements.txt
```

## 1.2. Download PlantCLEF2015 Dataset

We used PlantCLEF2016 (https://www.imageclef.org/lifeclef/2016/plant) as it contains all the data from PlantCLEF2015 in separate folders.

## 1.3. Download Pl@ntNet Dataset (Optional)

If you want to run dataset/stats.ipynb, you may want to download Pl@ntNet too (https://zenodo.org/records/4726653#.YhNbAOjMJPY).

## 1.4. Create Config

Copy example_config.json into the project root directory. Name it as config.json.

Specify there:

- **cache_dir:** our code may use it to store there files to speed up some operations.
- **plant_net_root:** path to the root directory of Pl@netNet dataset on your computer.
- **plant_clef_root:** path to the root directory of PlantCLEF2015 dataset on your computer.
- **plant_clef_train:** path to the directory with train split of PlantCLEF2015 dataset (relative to plant_clef_root).
- **plant_clef_test:** path to the directory with test split of PlantCLEF2015 dataset (relative to plant_clef_root).
- **working_dir:** path to the directory where our code will store various artefacts, e.g., models, logs, etc.

# 2. Project Structure

- **dataset** directory contains everything related to data:
    - **stats.ipynb** presents some dataset statistics.
    - **preprocessing.ipynb** contains our PlantCLEF2015 dataset preprocessing pipeline.
    - **loading.py** contains methods for loading datasets for our ML models.
    - **plant_net_meta.py** a model of Pl@ntNet metadata.
    - **plant_clef_meta.py** a model of PlantCLEF2015 metadata.
    - **data_loading_demo.ipynb** demonstration of using **loading.py**.
- **unimodal** directory contains everything related to our unimodal models:
    - **pretrain_modalities.py** a script to train and save weights of our unimodal models.
    - **weights_to_models.py** a script to convert weights of our unimodal models into keras models.
    - **__some_experiments** contains some of our experiments with unimodal models. Note that we included only a part of experiments, since others were messy :) Also, there may be some bugs in these versions of experiments.
- **multimodal** directory contains everything related to our multimodal model:
    - **mfas.py** contains our implementation of MFAS algorithm based on the original paper [(Perez-Rua et al., 2019)](https://www.researchgate.net/publication/338510163_MFAS_Multimodal_Fusion_Architecture_Search) and the author's [source code](https://github.com/jperezrua/mfas).
    - **final_model_fine_tuning.py** a script for training, fine-tuning and saving our final model.
    - **__some_experiments** contains some of our experiments with architectures found by MFAS. Note that we included only a part of experiments, since others were messy :) Also, there may be some bugs in these versions of experiments.
    - **classes** contains all the classes used in our MFAS implementation and in final model training, and in our experiments.
- **evaluation** contains the code used for model evaluation:
    - **evaluate_model.py** basic evaluation of unimodal models, our multimodal model or our baseline.
    - **mcnemar_test.py** McNemar's test to detect the statistical significance of difference between the proposed model and the baseline.
    - **missing_modalities.py** script to compare the final model with unimodal models and the baseline on subsets of modalities.
    - **evaluation_results** actual metrics of all our models.
    - **utils** various utilities for the evaluation.
- **common** various helpers and constants used in the project.
- **convert** utilities to convert models to other formats:
    - **convert_to_coreml.py** script to covert models to Apple CoreML format.
- **resources** various resources produced by our code:
    - **plant_clef_meta.json** metadata cached by **preprocessing.py** which we used in generation of our csv files – references to actual images.
    - **csv** references to actual images with class labels, used during model training, validation and evaluation.
    - **model** our trained models.

# 3. Execute Code

First start with **preprocessing.ipynb** to preprocess the data, and generate all the necessary files. Then you are ready to experiment with unimodal architectures, or multimodal architectures, including MFAS.

Execute all the code from the root project directory. Please check source files for expected parameters. For scripts that require modalities use _Flower Leaf Fruit Stem_, if a script also requires paths for the corresponding models, input them in the same order, for example:

```zsh
python -m evaluation.evaluate_model \
    --mode late_fusion \
    --modalities Flower Leaf Fruit Stem \
    --paths path/to/Flower/model.keras \
    path/to/Leaf/model.keras \
    path/to/Fruit/model.keras \
    path/to/Stem/model.keras
```
