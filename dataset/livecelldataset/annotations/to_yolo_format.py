from pycocotools.coco import COCO
import os
import json
import cv2

root = 'LIVECell_single_cells/'
types = ['a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'shsy5y', 'skbr3', 'skov3']

for split in ["train", "val", "test"]:
    annotations = COCO(f'livecell_coco_{split}.json')
    id_dict = {}
    with open(f"./images_txt/{split}.txt", "w") as f:
        for i in range(len(types)):
            file_names = []
            file_ids = []
            category_ids = []
            with open(os.path.join(root, types[i], f'{split}.json')) as json_f:
                anns = json.load(json_f)
                for ann in anns["images"]:
                    file_names.append(ann["file_name"])
                    file_ids.append(ann["id"])
                    category_ids.append(i + 1)
            
            for j in range(len(file_names)):
                file_name = file_names[j]
                if split == "test":
                    file_path = os.path.join('../dataset/livecelldataset_all/livecell_test_images', file_name)
                else:
                    file_path = os.path.join('../dataset/livecelldataset_all/livecell_train_val_images', file_name)
                f.write(f'{file_path}\n')
                file_id = file_ids[j]
                category_id = category_ids[j]
                id_dict[file_name] = file_id
                ann_ids = annotations.getAnnIds(
                            imgIds=[file_id], 
                            iscrowd=None
                        )
                anns = annotations.loadAnns(ann_ids)
                label_name = file_name.replace('.tif', '.txt')

                with open(os.path.join('./labels', label_name), 'w') as label_file:
                    for ann in anns:
                        x, y, w, h = ann['bbox']
                        x_center = (x + w / 2) / 704
                        y_center = (y + h / 2) / 520
                        w /= 704
                        h /= 520
                        label_file.write(f"{category_id} {x_center} {y_center} {w} {h}\n")
