TASK=4-1
CKPT_POSTFIX=50ep_batch32

STEP=0
python train.py --device 0 --batch-size 8 --img-size 640 640 --data livecell.yaml --exist-ok \
--cfg cfg/yolov4-csp-x-mish-cl.cfg --weights '' --name yolov4_${TASK}_step${STEP} \
--single-cls --workers 4 --task ${TASK} --load_task ${TASK} --step ${STEP} --ckpt_postfix ${CKPT_POSTFIX} \
--use_custom_img2label --use_segmentation

STEP=1
python train.py --device 0 --batch-size 8 --img-size 640 640 --data livecell.yaml --exist-ok \
--cfg cfg/yolov4-csp-x-mish-cl.cfg --weights '' --name yolov4_${TASK}_step${STEP} \
--single-cls --workers 4 --task ${TASK} --load_task ${TASK} --step ${STEP} --ckpt_postfix ${CKPT_POSTFIX} \
--use_custom_img2label --use_segmentation --epochs 80

STEP=2
python train.py --device 0 --batch-size 8 --img-size 640 640 --data livecell.yaml --exist-ok \
--cfg cfg/yolov4-csp-x-mish-cl.cfg --weights '' --name yolov4_${TASK}_step${STEP} \
--single-cls --workers 4 --task ${TASK} --load_task ${TASK} --step ${STEP} --ckpt_postfix ${CKPT_POSTFIX} \
--use_custom_img2label --use_segmentation --epochs 80

STEP=3
python train.py --device 0 --batch-size 8 --img-size 640 640 --data livecell.yaml --exist-ok \
--cfg cfg/yolov4-csp-x-mish-cl.cfg --weights '' --name yolov4_${TASK}_step${STEP} \
--single-cls --workers 4 --task ${TASK} --load_task ${TASK} --step ${STEP} --ckpt_postfix ${CKPT_POSTFIX} \
--use_custom_img2label --use_segmentation --epochs 80

STEP=4
python train.py --device 0 --batch-size 8 --img-size 640 640 --data livecell.yaml --exist-ok \
--cfg cfg/yolov4-csp-x-mish-cl.cfg --weights '' --name yolov4_${TASK}_step${STEP} \
--single-cls --workers 4 --task ${TASK} --load_task ${TASK} --step ${STEP} --ckpt_postfix ${CKPT_POSTFIX} \
--use_custom_img2label --use_segmentation --epochs 80
