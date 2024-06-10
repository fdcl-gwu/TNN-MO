from configparser import ConfigParser, ExtendedInterpolation
import json
from .loadargs import *
from .config import config2args
import os

def readargs(dir = ""):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser(dir)])
    args = parser.parse_args()
    # example = args.examplename
    # detr_version = args.detr_version
    config2args(args)
    print("config : ", args.configfile)
    return args

args = readargs()
config = ConfigParser()

configdirlist = config.read(args.configfile)
if len(configdirlist) == 0:
    print("l")
    config.read('config.ini')

cat_names = json.loads(config.get('Test', 'Obj_list'))
colors = json.loads(config.get('Test', 'COLORS'))
results_dir = config.get('Test', 'RESULTS')
example_dir = os.getcwd()+ config.get('Dir', 'EXAMPLES')