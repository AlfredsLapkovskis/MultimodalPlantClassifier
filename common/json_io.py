import os
import json
from shutil import copy2


class JsonIO:

    def __init__(
        self,
        json_path: str,
        merge_from_dir: str | None=None,
        pretty=False,
        backup_enabled=True,
    ):
        self._json_path = json_path
        self._merge_from_dir = merge_from_dir
        self._pretty = pretty
        self._backup_enabled = backup_enabled

            
    def read(self):
        if os.path.exists(self._json_path):
            return self._load_from_file()
        elif self._merge_from_dir is not None and os.path.exists(self._merge_from_dir):
            json_dict = self._load_from_merge_dir()
            if json_dict:
                self.write(json_dict)

        return None
    

    def write(self, json_dict):
        self._ensure_enclosing_dir_created()
        self._backup_if_needed()
        self._write(json_dict)
    

    def _load_from_file(self):
        with open(self._json_path, "r") as f:
            return json.loads(f.read())

    
    def _load_from_merge_dir(self):
        json_dict = dict()

        for path in os.listdir(self._merge_from_dir):
            if path.endswith(".json"):
                with open(os.path.join(self._merge_from_dir, path), "r") as f:
                    json_dict.update(json.loads(f.read()))
        
        return json_dict


    def _write(self, json_dict):
        with open(self._json_path, "w") as f:
            json_str = json.dumps(
                json_dict,
                indent=2 if self._pretty else None,
            )

            f.write(json_str)
    

    def _ensure_enclosing_dir_created(self):
        dir = os.path.dirname(self._json_path)
        os.makedirs(dir, exist_ok=True)

    
    def _backup_if_needed(self):
        if self._backup_enabled and os.path.exists(self._json_path):
            copy2(self._json_path, self._json_path + "_backup")
