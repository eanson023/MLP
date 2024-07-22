import os

import json
import orjson
from tqdm import tqdm
from sanitize_text import sanitize


def load_json(path):
    with open(path, "rb") as ff:
        return orjson.loads(ff.read())


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=2))


def save_dict_json(dico, path):
    # Sort the dictionary by keys
    dico = {k: v for k, v in sorted(dico.items())}
    write_json(dico, path)

def store_keyid(dico, keyid, path, duration, annotations):
    duration = round(float(duration), 3)

    # Create a dictionnary with all the info
    dico[keyid] = {
        "path": os.path.splitext(path)[0],
        "duration": duration,
        "annotations": []
    }

    for ann in annotations:
        start = ann.pop("start")
        end = ann.pop("end")
        element = ann.copy()

        element.update({
            "start": round(float(start), 3),
            "end": min(duration, round(float(end), 3))
        })

        dico[keyid]["annotations"].append(element)

def process_babel(babel_path: str, outputs: str = "datasets/babel", extra=False):
    os.makedirs(outputs, exist_ok=True)
    if not extra:
        save_json_index_path = os.path.join(outputs, "babel.json")

        train_path = os.path.join(babel_path, "train.json")
        val_path = os.path.join(babel_path, "val.json")
    else:
        save_json_index_path = os.path.join(outputs, "babel_extra.json")

        train_path = os.path.join(babel_path, "extra_train.json")
        val_path = os.path.join(babel_path, "extra_val.json")

    train_dico = load_json(train_path)
    val_dico = load_json(val_path)

    all_dico = {**val_dico, **train_dico}

    dico = {}
    for keyid, babel_ann in tqdm(all_dico.items()):
        path = babel_ann["feat_p"]
        babel_ann = all_dico[keyid]

        keyid = keyid.zfill(5)

        path = "/".join(path.split("/")[1:])
        dur = babel_ann["dur"]

        annotations = []
        
        labels = []
        if extra and babel_ann["frame_anns"] and babel_ann["frame_anns"][0]["labels"]:
            labels = babel_ann["frame_anns"][0]["labels"]
        elif not extra and babel_ann["frame_ann"] and babel_ann["frame_ann"]["labels"]:
            labels = babel_ann["frame_ann"]["labels"]

        for idx, data in enumerate(labels):
            text = data["raw_label"]
            text = sanitize(text)

            start = data["start_t"]
            end = data["end_t"]

            element = {
                # to save the correspondance
                # with the original BABEL dataset
                "seg_id": f"{keyid}_seg_{idx}",
                "babel_id": data["seg_id"],
                "text": text,
                "start": start,
                "end": end
            }

            annotations.append(element)

        # at least one
        if len(annotations) >= 1:
            store_keyid(dico, keyid, path, dur, annotations)

    # saving the annotations
    save_dict_json(dico, save_json_index_path)
    print(f"Saving the annotations to {save_json_index_path}")


if __name__ == "__main__":
    babel_path = "datasets/babel/babel_v1.0_release"
    process_babel(babel_path)
    process_babel(babel_path, extra=True)
