import json
import os


class Config:
    cache_dir: str
    plant_net_root: str
    plant_clef_root: str
    plant_clef_train: str
    plant_clef_test: str
    working_dir: str

    def get_log_dir(self):
        return os.path.join(self.working_dir, "logs")
    
    def get_checkpoint_dir(self):
        return os.path.join(self.working_dir, "checkpoint")
    
    def get_models_dir(self):
        return os.path.join(self.working_dir, "models")
    
    def get_trials_dir(self):
        return os.path.join(self.working_dir, "trials")

    def get_unimodal_file_path(self, split, modality):
        return os.path.join(self.plant_clef_root, "unimodal", split, f"{modality}.tfrecords")
    
    def get_multimodal_file_path(self, split):
        return os.path.join(self.plant_clef_root, "multimodal", split, "data.tfrecords")
    
    def get_unimodal_labels_file_path(self, split, modality):
        return os.path.join(self.plant_clef_root, "unimodal", split, f"{modality}.txt")
    
    def get_multimodal_labels_file_path(self, split):
        return os.path.join(self.plant_clef_root, "multimodal", split, "data.txt")

    
    def __init__(self, path):
        with open(path) as f:
            config = json.loads(f.read())

        self.cache_dir = config["cache_dir"]
        self.plant_net_root = config["plant_net_root"]
        self.plant_clef_root = config["plant_clef_root"]
        self.plant_clef_train = config["plant_clef_train"]
        self.plant_clef_test = config["plant_clef_test"]
        self.working_dir = config["working_dir"]
