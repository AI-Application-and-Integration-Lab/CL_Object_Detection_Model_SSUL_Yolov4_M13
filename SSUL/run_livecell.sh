DATA_ROOT=../dataset
DATASET=livecell
TASK=4-4 # 4-1
EPOCH=50
BATCH=32
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=100
CURR_STEP=0
CKPT_POSTFIX=${EPOCH}ep_batch${BATCH}

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0,1 --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --crop_size 512 --pseudo --pseudo_thresh ${THRESH} --unknown --norm_type gray \
    --unknown --w_transfer --amp --mem_size ${MEMORY} --curr_step ${CURR_STEP} --ckpt_postfix ${CKPT_POSTFIX} \
