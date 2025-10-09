#!/bin/bash

# Default values
TIME="24:00:00"
MEM="32G"
GPU="l40s"
NGPU=1
FOLD=0

# Parse args passed to sbatch
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        -m|--mem)
            MEM="$2"
            shift 2
            ;;
        -x|--gpu)
            GPU="$2"
            shift 2
            ;;
        -n|--ngpu)
            NGPU="$2"
            shift 2
            ;;
        -f|--fold)
            FOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

TMP_SCRIPT=$(mktemp)

cat <<EOF > $TMP_SCRIPT
#!/bin/bash
#SBATCH --account=aip-medilab
#SBATCH --job-name=lumpnav_medsam_train
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MEM}
#SBATCH --gres=gpu:${GPU}:${NGPU}
#SBATCH --output=/home/cyeung/projects/aip-medilab/cyeung/tracked-us-segmentation/logs/%x_%j.out
#SBATCH --error=/home/cyeung/projects/aip-medilab/cyeung/tracked-us-segmentation/logs/%x_%j.err

# ----------------------------
# Run training
# ----------------------------
mkdir -p /home/cyeung/projects/aip-medilab/cyeung/tracked-us-segmentation/logs

python train_one_gpu.py \
    -i /home/cyeung/projects/aip-medilab/cyeung/tracked-us-segmentation/data/LumpNavPatientArrays \
    -fold $FOLD \
    -num_epochs 100 \
    -batch_size 4 \
    -num_workers 2 \
    -use_wandb True \
    -wandb_project_name breast-tracked-us \

EOF

sbatch $TMP_SCRIPT
