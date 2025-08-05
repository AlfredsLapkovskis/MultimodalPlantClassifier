# Source code for the paper "Automatic fused multimodal deep learning for plant identification"

## Alfreds Lapkovskis, Natalia Nefedova & Ali Beikmohammadi (2025)

##### URL: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1616020/full

##### Please also check our sample iOS app that utilizes the proposed model (uses an older version of our model): https://github.com/AlfredsLapkovskis/MultimodalPlantClassifier-iOS

# Citation

Lapkovskis A, Nefedova N and Beikmohammadi A (2025) Automatic fused multimodal deep learning for plant identification. _Front. Plant Sci._ 16:1616020. doi: 10.3389/fpls.2025.1616020

BibTeX:
```
@article{lapkovskis16automatic,
  title={Automatic fused multimodal deep learning for plant identification},
  author={Lapkovskis, Alfreds and Nefedova, Natalia and Beikmohammadi, Ali},
  journal={Frontiers in Plant Science},
  volume={16},
  pages={1616020},
  year={2025},
  issn={1664-462X},
  publisher={Frontiers},
  doi={https://doi.org/10.3389/fpls.2025.1616020},
  url={https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1616020/full}
}
```

# 1. Setup

We used Python 3.11.5.

## 1.1. Configure a Virtual Environment

Execute these commands from the project root directory:

**Windows**:
```batch
python -m venv env
env/bin/activate
pip install -r requirements.txt
```

**Linux**:
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**MacOS:**
```zsh
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Optional step to accelerate training on MacOS (https://developer.apple.com/metal/tensorflow-plugin/)
pip install -r requirements_macos.txt
```

## 1.2. Download PlantCLEF2015 Dataset

We used PlantCLEF2016 (https://www.imageclef.org/lifeclef/2016/plant) as it contains all the data from PlantCLEF2015 in separate folders.

## 1.3. Download Pl@ntNet Dataset (Optional)

If you want to run dataset/stats.ipynb, you may want to download Pl@ntNet too (https://zenodo.org/records/4726653#.YhNbAOjMJPY).

## 1.4. Create Config

Copy example_config.json into the project root directory. Name it as **config.json**.

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
    - **preprocessing.ipynb** contains our pipeline to transform PlantCLEF2015 into Multimodal-PlantCLEF.
    - **generate_dataset.py** contains a script version of our pipeline to transform PlantCLEF2015 into Multimodal-PlantCLEF. Can be executed with custom parameters. See Section 4.
    - **loading.py** contains methods for loading datasets for our ML models.
    - **plant_net_meta.py** a model of Pl@ntNet metadata.
    - **plant_clef_meta.py** a model of PlantCLEF2015 metadata.
    - **data_loading_demo.ipynb** demonstration of using **loading.py**.
- **unimodal** directory contains everything related to our unimodal models:
    - **experiment.py** a class containing hyperparameters and settings for unimodal model training.
    - **experiments** a directory containing JSON files with experiment hyperparameters and settings. These files are parsed by **experiment.py** and must be named as **exp<i>\<index></i>.json**.
    - **train.py** a script to train our unimodal models. Usage: <code>python -m unimodal.train -e \<index of experiment></code>. Optionally, add <code>-s \<save mode></code> to save the trained models. For available save mode options, see: **common/save_mode.py**.
- **multimodal** directory contains everything related to our multimodal model:
    - **run_mfas.py** a script to run our implementation of MFAS algorithm based on the original paper [(Perez-Rua et al., 2019)](https://www.researchgate.net/publication/338510163_MFAS_Multimodal_Fusion_Architecture_Search) and the author's [source code](https://github.com/jperezrua/mfas).
    - **experiment.py** a class containing hyperparameters and settings for multimodal model training. We use it once we have found optimal configurations by **run_mfas.py**.
    - **experiments** a directory containing JSON files with experiment hyperparameters and settings. These files are parsed by **experiment.py** and must be named as **exp<i>\<index></i>.json**. We use it once we have found optimal configurations by **run_mfas.py**.
    - **train.py** a script to train our multimodal model. We use it once we have found an optimal configuration by **run_mfas**. Usage: <code>python -m unimodal.train -e \<index of experiment></code>. Optionally, add <code>-s \<save mode></code> to save the trained models. For available save mode options, see: **common/save_mode.py**.
    - **classes** contains all the classes used in our MFAS implementation (including **mfas.py**, the algorithm itself) and multimodal experiments.
- **evaluation** contains the code used for model evaluation:
    - **evaluate_model.py** basic evaluation of unimodal models, our multimodal model or our baseline.
    - **mcnemar_test.py** McNemar's test to detect the statistical significance of difference between the proposed model and the baseline.
    - **subsets_of_modalities.py** script to compare the final model with unimodal models and the baseline on subsets of modalities.
    - **utils** various utilities for the evaluation.
- **common** various helpers and constants used in the project.
- **convert** utilities to convert models to other formats:
    - **convert_to_coreml.py** script to covert models to Apple CoreML format.
- **resources** various resources produced by our code:
    - **models** our trained models. **train** directory contains models trained on training set only, whereas **train+validation** contains models trained on merged training and validation sets.

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

# 4. Generate Multimodal-PlantCLEF

To generate Multimodal-PlantCLEF, please follow instructions in Section 1.1, 1.2 and 1.4. Then use our **generate_dataset.py** script:

```zsh
python -m dataset.generate_dataset
```

Optionally, specify custom parameters:

```zsh
python -m dataset.generate_dataset --help                                                             


usage: generate_dataset.py [-h] [--verbose VERBOSE] [--modalities MODALITIES [MODALITIES ...]] [--split_dist SPLIT_DIST [SPLIT_DIST ...]] [--max_solver_iter MAX_SOLVER_ITER] [--seed SEED]
                           [--plot_splits PLOT_SPLITS] [--dims DIMS [DIMS ...]]

options:
  -h, --help            show this help message and exit
  --verbose VERBOSE     Enable additional logging
  --modalities MODALITIES [MODALITIES ...]
                        Specify modalities to include into the dataset (Leaf, Flower, Fruit, Stem, Bark, Branch, Scan, ...)
  --split_dist SPLIT_DIST [SPLIT_DIST ...]
                        Specify train, validation and test split sizes (fractions in [0, 1])
  --max_solver_iter MAX_SOLVER_ITER
                        Limit solver iterations to reduce computational time
  --seed SEED           Random seed, for reproducibility of results
  --plot_splits PLOT_SPLITS
                        Plot data split distributions and save them into `images` directory
  --dims DIMS [DIMS ...]
                        Saved image dimensions
```

This will generate folders with unimodal and multimodal datasets (.tfrecords files), splitted into train, validation and test splits at your **plant_clef_root** location (see Section 1.4). Data instances in the dataset contain "image" and "label" properties. Please refer to **dataset/loading.py** for an example, how to load images and labels. For convenience, there will also be .txt files with labels corresponding to data instances in the datasets.
