import os
from glob import glob
from tqdm import tqdm


def loop_amass(
    base_folder,
    new_base_folder,
    ext=".npz",
    newext=".npz",
    force_redo=False,
    exclude=None,
):
    match_str = os.path.join(base_folder, f"**/*{ext}")

    for motion_path in tqdm(glob(match_str, recursive=True)):
        if exclude and exclude in motion_path:
            continue

        # motion_path = os.path.join(base_folder, motion_file)

        if motion_path.endswith("shape.npz"):
            continue

        new_motion_path = motion_path.replace(base_folder, new_base_folder).replace(ext, newext)
        
        if not force_redo and os.path.exists(new_motion_path):
            continue

        new_folder = os.path.split(new_motion_path)[0]
        os.makedirs(new_folder, exist_ok=True)

        yield motion_path, new_motion_path
