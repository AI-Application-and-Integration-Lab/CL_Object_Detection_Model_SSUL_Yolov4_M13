TASK=4-4
STEP=1
DATASET=livecell
CKPT_POSTFIX=50ep_batch32

python test.py --img-size 640 --conf 0.001 --batch 8 --device 0 \
--data livecell.yaml --cfg cfg/yolov4-csp-x-mish-cl.cfg --weights runs/train/yolov4_${TASK}_step${STEP}/weights/best.pt --ckpt_postfix ${CKPT_POSTFIX} \
--task offline --load_task ${TASK} --ssul_root ../SSUL --step 0 --name ${TASK}_step${STEP} --names data/livecell.names --use_custom_img2label \
--task_stage test --exist-ok --dataset $DATASET --use_segmentation
