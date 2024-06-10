EXT=".png"
ZN=0  #images with zero numbering. eg. 000001.  True 1 False 0 
GT=0  #images with groundtruth. True 1 False 0 

CKPTFOLDER="TNN-MO"
CONFIG_DIR="checkpoints/TNN_MO_6-Object_model/"
CONFIG=$CONFIG_DIR"config.ini"
python3 scripts/inference.py --configfile $CONFIG  --ext $EXT --zero_numbering $ZN --gt $GT 
#python3 scripts/plotTraj.py --configfile $CONFIG 
