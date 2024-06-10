
def build_model(args):
    
    if args.detr_version=='1.0': #keypoints out dim 1x96 
        from .detr1 import build 

    if args.detr_version=='1.1': #keypoints out dim 1x64 
        from .detr1_1 import build 

    return build(args)

        

