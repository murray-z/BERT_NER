import json


def load_json(json_file_path):
    with open(json_file_path) as f:
        return json.loads(f.read())


def dump_json(obj, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)



        