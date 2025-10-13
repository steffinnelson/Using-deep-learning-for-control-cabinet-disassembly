import os
import shutil
import random
import imagehash
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import pickle
import csv
import subprocess
import re

def run_dataset_reduction(config):
    SOURCE_DIR = config["reducer_source_dir"]
    DEST_DIR = config["reducer_dest_dir"]
    YOLO_DATA = config["data_yaml"]
    BATCH_SIZE = config.get("batch_delete_size", 50)
    DELETE_EVAL_EPOCHS = config.get("delete_eval_epochs", 20)
    MIN_TRAIN_IMAGES = config.get("min_train_images", 100)
    TARGET_CLASSES = set(config.get("required_classes", []))
    MAX_DROP_PERCENT = config.get("max_map95_drop_percent", 5)
    target_thresholds = config.get("TARGET_MAP95_THRESHOLDS", {})

    RECOVERY_IMAGES = config["test_images_dir"]
    RECOVERY_LABELS = config["test_labels_dir"]
    CSV_LOG = "reduction_metrics_batchwise.csv"
    RETAIN_IMAGES = set(os.path.basename(path) for path in config.get("retain_images", []))

    os.makedirs(RECOVERY_IMAGES, exist_ok=True)
    os.makedirs(RECOVERY_LABELS, exist_ok=True)

    shutil.copytree(SOURCE_DIR, DEST_DIR, dirs_exist_ok=True)
    print(f"✅ Copied dataset to reduced folder: {DEST_DIR}")

    SRC_IMAGES = os.path.join(DEST_DIR, "images")
    SRC_LABELS = os.path.join(DEST_DIR, "labels")

    print("🔎 Computing image hashes for similarity scoring...")
    hash_dict = {}
    for f in tqdm(os.listdir(SRC_IMAGES)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = Image.open(os.path.join(SRC_IMAGES, f)).convert("RGB")
                hash_dict[f] = imagehash.average_hash(img)
            except Exception as e:
                print(f"⚠️ Error reading image {f}: {e}")

    with open("image_hashes.pkl", "wb") as f:
        pickle.dump(hash_dict, f)

    deletable = []
    annotation_counts = defaultdict(int)

    for label_file in os.listdir(SRC_LABELS):
        if not label_file.endswith('.txt'):
            continue

        base = os.path.splitext(label_file)[0]
        image_file_candidates = [base + ext for ext in ['.jpg', '.jpeg', '.png']]
        if any(img in RETAIN_IMAGES for img in image_file_candidates):
            continue  # ❌ Skip if image is marked for retention

        path = os.path.join(SRC_LABELS, label_file)
        with open(path, "r") as f:
            lines = f.readlines()

        anns = defaultdict(int)
        image_classes = set()
        for line in lines:
            cls = int(line.strip().split()[0])
            anns[cls] += 1
            image_classes.add(cls)

        for cls in anns:
            annotation_counts[cls] += anns[cls]

        if image_classes.issubset(TARGET_CLASSES):
            deletable.append((label_file, anns))

    print(f"🧹 {len(deletable)} deletable candidates found.")

    # === Sort by similarity ===
    def similarity_score(fname):
        others = [h for f, h in hash_dict.items() if f != fname]
        return sum(abs(hash_dict[fname] - h) for h in others)

    deletable.sort(key=lambda item: similarity_score(item[0].replace('.txt', '.jpg')))

    batch_id = 0
    previous_map95 = {}

    with open(CSV_LOG, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "deleted_files"] + [f"cls_{c}_count" for c in sorted(TARGET_CLASSES)])

    while deletable:
        batch_id += 1
        this_batch = deletable[:BATCH_SIZE]
        deletable = deletable[BATCH_SIZE:]

        print(f"\n🚧 Batch {batch_id}: Deleting {len(this_batch)} images...")
        moved_files = []

        for label_file, anns in this_batch:
            base = os.path.splitext(label_file)[0]
            img_file = next((base + ext for ext in ['.jpg', '.jpeg', '.png']
                             if os.path.exists(os.path.join(SRC_IMAGES, base + ext))), None)

            src_lbl = os.path.join(SRC_LABELS, label_file)
            dst_lbl = os.path.join(RECOVERY_LABELS, label_file)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

            if img_file:
                src_img = os.path.join(SRC_IMAGES, img_file)
                dst_img = os.path.join(RECOVERY_IMAGES, img_file)
                if os.path.exists(src_img):
                    shutil.move(src_img, dst_img)
                    moved_files.append((img_file, label_file))

            for cls in anns:
                annotation_counts[cls] -= anns[cls]

        snapshot_dir = f"snapshot_batch_{batch_id}"
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        shutil.copytree(DEST_DIR, snapshot_dir)

        print(f"🧪 Validating after deletion (epochs={DELETE_EVAL_EPOCHS})...")
        train_cmd = f"yolo train data={YOLO_DATA} model=yolo11l.pt imgsz= 1280 epochs={DELETE_EVAL_EPOCHS} name=batch_{batch_id}"
        val_cmd = f"yolo val model=runs/detect/batch_{batch_id}/weights/best.pt data={YOLO_DATA}"

        os.system(train_cmd)
        result = subprocess.run(val_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        val_output = result.stdout
        print(val_output)

        rollback = False
        for cls in sorted(TARGET_CLASSES):
            cls_str = str(cls)
            for line in val_output.splitlines():
                if line.strip().startswith(cls_str):
                    parts = re.findall(r"[\d.]+", line)
                    if len(parts) >= 6:
                        current = float(parts[5])
                        prev = previous_map95.get(cls, current)
                        drop = 100 * (prev - current) / (prev + 1e-6)
                        abs_thresh = float(target_thresholds.get(cls_str, 0))  # default to 0 if not defined

                        print(f"🔎 Class {cls} mAP95: {current:.3f} (↓ {drop:.1f}%) [Min: {abs_thresh}]")

                        if drop > MAX_DROP_PERCENT:
                            print(f"🛑 Class {cls} mAP95 dropped >{MAX_DROP_PERCENT}% → rollback.")
                            rollback = True
                            break

                        if current < abs_thresh:
                            print(f"🛑 Class {cls} mAP95 below absolute threshold ({abs_thresh}) → rollback.")
                            rollback = True
                            break
                    break
            if rollback:
                break

        if rollback:
            for img_file, label_file in moved_files:
                shutil.move(os.path.join(RECOVERY_IMAGES, img_file), os.path.join(SRC_IMAGES, img_file))
                shutil.move(os.path.join(RECOVERY_LABELS, label_file), os.path.join(SRC_LABELS, label_file))
            break

        for cls in sorted(TARGET_CLASSES):
            previous_map95[cls] = current

        print(f"✅ Batch {batch_id} accepted.")

        with open(CSV_LOG, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([batch_id, len(this_batch)] + [annotation_counts[c] for c in sorted(TARGET_CLASSES)])

    print("✅ Reduction complete.")
    return True
