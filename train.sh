CONFIG_DIR=""

Resume="OFF"

if [ "$Resume" = "OFF" ]; then
    CONFIG_DIR=""
fi

if [ "$Resume" = "ON" ]; then
    CKPT="/TNN-MO/checkpoints/"
    CONFIG_DIR="TNN_MO_6-Object_model"
fi


if [ "$Resume" = "OFF" ]; then
    CONFIG=$CONFIG_DIR"config.ini"
    echo "\033[36mConverting pascal format ship data to coco format\033[0m"
    python3 scripts/ship_to_coco.py --configfile $CONFIG
    echo "\033[36mSTART training .......\033[0m"
    python3 scripts/main.py --configfile $CONFIG
fi

if [ "$Resume" = "ON" ]; then
    echo "\033[36mSTART training .......\033[0m"
    CONFIG=$CKPT$CONFIG_DIR"/config.ini"
    python3 scripts/main.py --configfile $CONFIG
fi


