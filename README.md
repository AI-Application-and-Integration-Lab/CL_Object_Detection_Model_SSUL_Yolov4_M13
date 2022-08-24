# CL_Object_Detection_Model_SSUL_Yolov4_M13

## Installation
SSUL:
```
# create conda environment
conda create -n ssul python=3.8
conda activate ssul
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r SSUL/requirements.txt
```

Yolov4:
```
# create conda environment
conda create -n yolov4 python=3.8
conda activate yolov4
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r yolov4/requirements.txt
```

## Data preparation
Download [LIVECell](https://github.com/sartorius-research/LIVECell) dataset, and place them into `dataset` folder like the following structure. 
```
.
├── SSUL
├── yolov4
├── dataset
│   ├── livecelldataset
│   │   └── annotations
│   │       |── images_txt
|   |       |── labels
|   |       |── to_yolo_format.py
|   |       |── livecell_coco_train.json
|   |       |── livecell_coco_val.json
|   |       |── livecell_coco_test.json
|   |       └── LIVECell_single_cells
│   │           ├── a172
│   │           |   ├── train.json
│   │           |   ├── val.json
│   │           |   └── test.json
│   │           ├── bt474
│   │           ├── bv2
│   │           ├── huh7
│   │           ├── mcf7
│   │           ├── shsy5y
│   │           ├── skbr3
│   │           └── skov3
│   └── livecelldataset_all
|       ├── livecell_train_val_images
│       │   ├── xxx.tif
|       |   └── ...
│       └── livecell_test_images
│           ├── xxx.tif
|           └── ...
└── ...
```
After placing files in the correct paths, run `python to_yolo_format.py` in folder `annotations` to generate input files for yolov4. Generated files are located in `images_txt` and `labels`.

## Training
SSUL:
```
cd SSUL
# you can check all arguments in SSUL/main.py
# change TASK argument in sh file to train different setting
bash run_livecell.sh

# you can check all arguments in SSUL/eval.py
# change TASK argument in sh file to test different setting
# add --save_mask to generate semantic segmentation results for yolov4 (when --save_mask is on, cropping will not be used)
bash run_livecell_test.sh
```

Yolov4:
```
cd yolov4
# you can check all arguments in yolov4/train.py
# for training setting 4-4
bash train_4-4.sh
# for training setting 4-1
bash train_4-1.sh
```

## Testing
SSUL:
```
cd SSUL
# you can check all arguments in SSUL/eval.py
# change TASK argument in sh file to test different setting
# remove --save_mask to only evaluate the test set performance
bash run_livecell_test.sh
```

Yolov4:
```
cd yolov4
# you can check all arguments in yolov4/test.py
# change TASK argument in sh file to test different setting
bash test.sh
```

## Acknowledgement
Our implementation is based on these repositories: [SSUL](https://github.com/clovaai/SSUL), [Yolov4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
