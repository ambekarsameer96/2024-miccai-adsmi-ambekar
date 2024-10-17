def compute_dice(predictions, gt, th):
    predictions[predictions>th]=1
    predictions[predictions<1]=0
    eps = 1e-6
    
    inputs = predictions.flatten()
    targets = gt.flatten()

    intersection = (inputs * targets).sum()
    dice = (2. * intersection) / (inputs.sum() + targets.sum() + eps)
    return dice


def compute_dice_tp_fp(predictions, gt, th):
    predictions[predictions > th] = 1
    predictions[predictions <= th] = 0
    eps = 1e-6
    
    inputs = predictions.flatten()
    targets = gt.flatten()

    
    intersection = (inputs * targets).sum()
    TP = (inputs * targets).mean()

    
    FP = (inputs * (1 - targets)).mean()

    
    dice = (2. * intersection) / (inputs.sum() + targets.sum() + eps)

    return dice, TP, FP
