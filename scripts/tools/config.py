from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import os

def setargs2config(args):
    try:
        os.mkdir(args.checkpoint_dir)
    except:
        pass

    try:
        os.mkdir(args.checkpoint_dir+args.testname + "/")
    except:
        pass
    
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.optionxform = lambda option: option.upper()

    config['Test'] = {'NUM_QUERIES': args.num_queries,
                      'NUM_CLASSES' : args.num_classes,
                      'OBJ_LIST' : args.Obj_list,
                      'COLORS' : args.colors,
                      'TESTNAME': args.testname,
                      'SNAPSHOT' : args.testname +'.pth',
                      'RESULTS' : args.results}
    
    config['Dir'] = {'DATA_DIR': args.data_dir,
                      'COCO_PATH': args.coco_path,
                      'COCOSUBFOLDER' : args.subfolder,
                      'CKPT': os.getcwd()+"/" + args.checkpoint_dir,
                      'EXAMPLES': args.examples}
    
    config['Training'] = {'DETR_VERSION': args.detr_version,
                      'RESUME': "ON",
                      'DIST': args.dist,
                      'DATASET_SIZE' : args.dataset_size,
                      'EPOCHS': args.epochs,
                      'BATCH_SIZE': args.batch_size}
    
    config['DETR'] = {'NUM_KEYS': args.num_keys,
                      'HEADS': args.nheads,
                      'ELAYERS': args.enc_layers,
                      'DLAYERS': args.dec_layers,
                      'DIM_FEEDFORWARD' : args.dim_feedforward,
                      'HIDDEN_DIM': args.hidden_dim,
                      'AUXLOSS': args.aux_loss}
    print("ch : ",args.checkpoint_dir+args.testname)
    with open(args.checkpoint_dir+args.testname+'/config.ini', 'w') as configfile:
        config.write(configfile)

    return config

def saveconfig(args, config, startTime, epoch):
    now = datetime.now()
    diff = now - startTime
    config['Training'] = {'Training_Start_Time': startTime,
                        'Trained_Time': now,
                        'Trained_Days' : diff.days,
                        'epochs': epoch+1,
                        'snapshot' : args.testname +'.pth'}
    
    with open(args.checkpoint_dir+args.testname+'/pth_config.ini', 'w') as configfile:
        config.write(configfile)

def config2args(args):
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.optionxform = lambda option: option.upper()

    configdirlist = config.read(args.configfile)
    if len(configdirlist) == 0:
        W = '\033[43m' + " Warning >>" + '\033[0m'
        print( W + '\033[100m' + " reading default config.ini" + '\033[0m')
        config.read('config.ini')

    args.checkpoint_dir = config.get('Dir', 'CKPT')
    # try:
    #     if args.RevNo == 0:
    #         os.mkdir(checkpoint_dir+"E"+args.ExpNo)
    #         print('Created Experiment E', args.ExpNo, ' folder ...')
            
    #     else:
    #         os.mkdir(checkpoint_dir+"E"+args.ExpNo+"_rev"+str(args.RevNo))
    #         print('Created Experiment E', args.ExpNo + "_rev" + str(args.RevNo), ' folder ...')
    #         shutil.copy2(file, checkpoint_dir+"E"+args.ExpNo+"_rev"+str(args.RevNo)+"/config.ini")
    #         print('Use exisiting config from ', args.ExpNo, ' folder ...')
    # except:
    #     if args.RevNo == 0:
    #         config.read(checkpoint_dir+"E"+args.ExpNo+"/config.ini")
    #         args.coco_path = checkpoint_dir

    #     else:
    #         config.read(checkpoint_dir+"E"+args.ExpNo+"_rev"+str(args.RevNo)+"/config.ini")
            
    #shutil.copy2(file, checkpoint_dir+"/config.ini")

    args.checkpoint = config.get('Test', 'TESTNAME')
    args.snapshot = config.get('Test', 'SNAPSHOT')
    args.testname = config.get('Test', 'TESTNAME')
    args.Obj_list = config.get('Test', 'OBJ_LIST')
    args.colors = config.get('Test', 'COLORS')
    args.results = config.get('Test', 'RESULTS')
    args.num_queries = config.getint('Test', 'NUM_QUERIES')
    args.num_classes = config.getint('Test', 'NUM_CLASSES')

    args.data_dir = os.getcwd() + config.get('Dir', 'DATA_DIR')
    args.coco_path = config.get('Dir', 'COCO_PATH')
    args.subfolder = config.get('Dir', 'COCOSUBFOLDER')
    args.examples = config.get('Dir', 'EXAMPLES')

    args.detr_version = config.get('Training', 'DETR_VERSION')
    args.resume = config.get('Training', 'RESUME')
    args.dist= config.get('Training', 'DIST')
    try:
        args.device = config.getint('Training', 'DEVICE')
    except:
        args.device = 'cuda:0'
    print("args.device : ", args.device)
    args.dataset_size = config.getint('Training', 'DATASET_SIZE')
    args.epochs = config.getint('Training', 'EPOCHS')
    args.batch_size = config.getint('Training', 'BATCH_SIZE')
    
    args.num_keys = config.getint('DETR', 'NUM_KEYS')
    args.nheads = config.getint('DETR', 'HEADS')
    args.enc_layers = config.getint('DETR', 'ELAYERS')
    args.dec_layers = config.getint('DETR', 'DLAYERS')
    args.dim_feedforward = config.getint('DETR', 'DIM_FEEDFORWARD')
    args.hidden_dim = config.getint('DETR', 'HIDDEN_DIM')
    args.aux_loss = config.getboolean('DETR', 'AUXLOSS')

    return args

