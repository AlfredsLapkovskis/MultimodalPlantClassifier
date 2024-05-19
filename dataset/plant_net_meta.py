import os
import json
from config import Config
from dataclasses import dataclass


@dataclass
class PlantNetImage:
    species_id: str
    obs_id: str
    organ: str
    is_shareable: bool
    v1_id: str
    author: str
    license: str
    split: str

    @staticmethod
    def load(config: Config) -> dict[str:]:
        meta_path = os.path.join(config.plant_net_root, "plantnet300K_metadata.json")

        with open(meta_path) as f:
            meta = {}

            for key, value in json.loads(f.read()).items():
                meta[key] = PlantNetImage(
                    species_id=value["species_id"],
                    obs_id=value["obs_id"],
                    organ=value["organ"],
                    is_shareable=value["is_shareable"],
                    v1_id=value["v1_id"],
                    author=value["author"],
                    license=value["license"],
                    split=value["split"],
                )

        return meta