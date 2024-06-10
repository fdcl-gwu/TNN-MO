
CONFIG=$CKPT$CONFIG_DIR"/config.ini"
CONFIG=$CONFIG_DIR"config.ini"
python3 scripts/ship_to_coco.py --configfile $CONFIG
torchrun --standalone --nproc_per_node=gpu scripts/main.py --configfile $CONFIG