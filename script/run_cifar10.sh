#!/bin/bash
set -e 

GPUS_STAGE1="0,1"
GPUS_STAGE2="0"
GPUS_STAGE3="0"
WORKERS=4

DATA_DIR="../datasets/CIFAR"
OUTPUT_ROOT="../output_dir"

EXP_NAME="cifar10_lr002_t05"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
DATASET="CIFAR10"
ARCH="resnet18"
SEED=1228

GMM_PERCENTILE=0.1
HYBRID_RATIO=0.5

# ================== Stage 1: Modeling the Visual Pattern Distribution ==================
CUDA_VISIBLE_DEVICES=${GPUS_STAGE1} python ../stage1_modeling.py \
  --dist-url tcp://localhost:10003 --multiprocessing-distributed --world-size 1 --rank 0 \
  --arch ${ARCH} --dataset ${DATASET} --img_size 224 \
  --data_dir ${DATA_DIR} \
  --exp_dir ${EXP_DIR} \
  --lr 0.02 --epochs 30 --batch_size 128 \
  --mlp --aug-plus --cos --moco-t 0.5

STAGE1_CKPT="${EXP_DIR}/checkpoint_0030.pth.tar"
TEACHER_CHECKPOINT="../checkpoints/resnet18_cifar10_200epochs.pth"

for IPC in 10 50 100; do
  DISTILLED_DIR_NAME="distilled_dataset_ipc${IPC}_k${GMM_PERCENTILE}_hybrid_ratio${HYBRID_RATIO}"

  # ================== Stage 2: Pattern-Balanced Coreset Selection ==================
  CUDA_VISIBLE_DEVICES=${GPUS_STAGE2} python ../stage2_selection.py \
    --dist-url tcp://localhost:10004 --multiprocessing-distributed --world-size 1 --rank 0 \
    --resume ${STAGE1_CKPT} \
    --dataset ${DATASET} --data_dir ${DATA_DIR} --arch ${ARCH} \
    --exp_dir ${EXP_DIR} \
    --ipc ${IPC} --seed ${SEED} \
    --entropies-path ./CIFAR10_train_entropies.npy \
    --gmm-uncertainty-percentile ${GMM_PERCENTILE} \
    --hybrid-gmm-centroid-ratio ${HYBRID_RATIO}

  # ================== Stage 3: Distillation Training ==================
  CUDA_VISIBLE_DEVICES=${GPUS_STAGE3} python ../stage3_training.py \
    --dataset ${DATASET} \
    --ipc ${IPC} \
    --distilled-data-path "${EXP_DIR}/${DISTILLED_DIR_NAME}" \
    --real-data-path ${DATA_DIR} \
    --teacher-resume ${TEACHER_CHECKPOINT} \
    --teacher-arch ${ARCH} \
    --student-arch ${ARCH} \
    --output-dir "../distilled_training_results/${EXP_NAME}/${DISTILLED_DIR_NAME}" \
    --epochs 500 \
    --lr 0.001 --wd 0.01 \
    --workers ${WORKERS} \
    --gpu 0 \
    --mix-type "cutmix" --temperature 20.0 --num-runs 3

done

echo "All Stages Completed Successfully!"