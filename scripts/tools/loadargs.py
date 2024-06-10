import argparse
import os
# parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()
# example = args.examplename
# detr_version = args.detr_version

def get_args_parser(dir = ""):
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--detr_version', default='1.1', type=str,
                        help="Name of the detr")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=2, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_classes', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    
    parser.add_argument('--num_keys',default=32, type=int,help="Number of keypoints")
    parser.add_argument('--points_only',default=0, type=int,help="Consider only points not a box")
    
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_keypoints', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--dataset_size', default=1000, type=int)
    parser.add_argument('--coco_path', default=os.getcwd() + '/synthetic_dataset', type=str)
    parser.add_argument('--main_dir', default=os.getcwd() + '/', type=str)
    parser.add_argument('--data_dir', default=os.getcwd() + '/synthetic_dataset/offline_saved/', type=str)
    parser.add_argument('--subfolder',default='small', type=str)
    parser.add_argument('--testname',default='test_00', type=str)
    parser.add_argument('--examplename',default='1', type=str)
    parser.add_argument('--resultname',default='1', type=str)
    parser.add_argument('--configfile',default=dir+'config.ini', type=str)
    # Intrinsic camera parameter folder
    parser.add_argument('--K_dir', default=os.getcwd() + '/synthetic_dataset/cam_prop', type=str)

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--checkpoint_dir', default=os.getcwd() + '/snapshots/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint', default=os.getcwd() + 'checkpoint.pth',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained', default='', help='load pretrained detr enc-dec and mm backend')
    parser.add_argument('--resume', default="OFF", type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--dist', default="OFF", type=str)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--ExpNo', default="1", type=str)
    parser.add_argument('--RevNo', default="0", type=str)
    parser.add_argument('--zero_numbering', default=True, type=int)
    parser.add_argument('--gt', default=False, type=int)
    parser.add_argument('--ext', default=".jpg", type=str)
    # # Add an argument for the list of integers
    # parser.add_argument('--catID_list', default= [1,2,3,4,5,6,7,8,9,0], type=list_of_catIDs) 
 
    return parser

def args_f():
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    return args

# # Define a custom argument type for a list of integers
def list_of_catIDs(arg):
    return list(map(int, arg.split(',')))

