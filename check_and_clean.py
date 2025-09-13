# check_and_clean.py
import os
from PIL import Image
from pathlib import Path
import shutil

root = Path("waste_raw")  
bad_dir = Path("bad_images")
bad_dir.mkdir(exist_ok=True)

def fix_image(p: Path):
    try:
        im = Image.open(p)
        im.verify()  
        im = Image.open(p).convert("RGB")
        new_p = p.with_suffix('.jpg')
        if p.suffix.lower() != '.jpg':
            im.save(new_p, "JPEG", quality=90)
            if new_p != p:
                p.unlink(missing_ok=True)
        return True
    except Exception as e:
        print("Bad:", p, "->", e)
        shutil.move(str(p), str(bad_dir / p.name))
        return False

for root_dir, dirs, files in os.walk(root):
    for fname in files:
        p = Path(root_dir) / fname
        fix_image(p)
print("Done!")