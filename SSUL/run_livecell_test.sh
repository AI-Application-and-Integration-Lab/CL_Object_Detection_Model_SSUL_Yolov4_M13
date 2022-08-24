DATA_ROOT=../../dataset
DATASET=livecell
TASK=4-4 # 4-1
EPOCH=50
BATCH=1 # inference batch size
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=0
CKPT_POSTFIX=${EPOCH}ep_batch32

python eval.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0,1 --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} \
    --unknown --w_transfer --amp --mem_size ${MEMORY} --ckpt_postfix ${CKPT_POSTFIX} --test_only \
    --crop_size 512 --crop_val --norm_type gray --save_mask
