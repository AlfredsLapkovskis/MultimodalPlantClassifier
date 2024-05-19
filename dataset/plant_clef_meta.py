import os
import re
import json
import xml.etree.ElementTree as ET
from config import Config
from dataclasses import dataclass


@dataclass
class PlantClefImage:
    observation_id: int
    media_id: int
    vote: float
    class_id: int
    content: str
    family: str
    genus: str
    species: str
    author: str
    date: str
    location: str
    latitude: float
    longitude: float
    year_in_clef: str
    observation_id_2014: int
    image_id_2014: int
    learn_tag: str


    def get_image_file_path(self, config: Config):
        return os.path.join(
            config.plant_clef_root,
            config.plant_clef_train if self.learn_tag == "Train" else config.plant_clef_test,
            f"{self.media_id}.jpg"
        )


    @staticmethod
    def load(config: Config, from_cache=False):
        if from_cache:
            meta = PlantClefImage.__load_from_cache(config)
            if meta is not None:
                return meta

        return PlantClefImage.__load_from_dataset(config)
    
    
    @staticmethod
    def save(config: Config, meta, overwrite=False, pretty=False):
        location = PlantClefImage.__get_cache_file(config)
        if os.path.exists(location) and not overwrite:
            return
        os.makedirs(os.path.dirname(location), exist_ok=True)

        meta_dicts = [vars(m) for m in meta]
        json_str = json.dumps(meta_dicts, indent=2 if pretty else None)

        with open(location, "w") as f:
            f.write(json_str)


    @staticmethod
    def __load_from_dataset(config: Config):
        meta = []

        for dir_name in [config.plant_clef_train, config.plant_clef_test]:
            dir_path = os.path.join(config.plant_clef_root, dir_name)

            for dir, _, files in os.walk(dir_path):
                for file in files:
                    m = re.match(r"^\d+\.xml$", file)
                    if m:
                        file_path = os.path.join(dir, file)
                        tree = ET.parse(file_path)
                        root = tree.getroot()

                        observation_id_2014 = root.find("ObservationId2014").text
                        image_id_2014 = root.find("ImageId2014").text

                        meta.append(PlantClefImage(
                            observation_id=int(root.find("ObservationId").text),
                            media_id=int(root.find("MediaId").text),
                            vote=float(root.find("Vote").text),
                            content=root.find("Content").text,
                            class_id=int(root.find("ClassId").text),
                            family=root.find("Family").text,
                            genus=root.find("Genus").text,
                            species=root.find("Species").text,
                            author=root.find("Author").text,
                            date=root.find("Date").text,
                            location=root.find("Location").text,
                            latitude=root.find("Latitude").text,
                            longitude=root.find("Longitude").text,
                            year_in_clef=root.find("YearInCLEF").text,
                            observation_id_2014=int(observation_id_2014) if observation_id_2014 else None,
                            image_id_2014=int(image_id_2014) if image_id_2014 else None,
                            learn_tag=root.find("LearnTag").text,
                        ))

        return meta


    @staticmethod
    def __load_from_cache(config):
        cache_location = PlantClefImage.__get_cache_file(config)
        if not os.path.exists(cache_location):
            return None
        
        with open(cache_location, "r") as f:
            json_str = f.read()

            return json.loads(
                json_str,
                object_hook=lambda d: PlantClefImage(**d),
            )


    @staticmethod
    def __get_cache_file(config: Config):
        return os.path.join(config.cache_dir, "plant_clef_meta.json")
