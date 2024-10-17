import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt


from neural_style.models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16Experimental
import copy 
import wandb 
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]






def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  

    if target_shape is not None:  
        if isinstance(target_shape, int) and target_shape != -1:  
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    
    img = img.astype(np.float32)  
    img /= 255.0  
    return img


def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)

    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0)

    return img


def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])  


def generate_out_img_name(config):
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    
    if 'reconstruct_script' in config:
        suffix = f'_o_{config["optimizer"]}_h_{str(config["height"])}_m_{config["model"]}{config["img_format"][1]}'
    else:
        suffix = f'_o_{config["optimizer"]}_i_{config["init_method"]}_h_{str(config["height"])}_m_{config["model"]}_cw_{config["content_weight"]}_sw_{config["style_weight"]}_tv_{config["tv_weight"]}{config["img_format"][1]}'
    return prefix + suffix


def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations, should_display=False, content_img_copy=None,style_img_copy=None):
    saving_freq = config['saving_freq']
    
    opt_img = copy.deepcopy(optimizing_img)
    
    anomaly_map, anomaly_score = get_anomaly(opt_img,content_img_copy)
    
    
    
    
    
    
    
    
    
     
     
    
    
    
    

    
    
    
    
    
    cnt_img = content_img_copy.squeeze(axis=0).to('cpu').detach().numpy()
    
    content_img_copy = content_img_copy.squeeze(axis=0).to('cpu').detach().numpy()
    
    
    content_img_copy = img_process(content_img_copy)
    
    
    style_img_copy = style_img_copy.squeeze(axis=0).to('cpu').detach().numpy()
    style_img_copy = img_process(style_img_copy)
    
    anomaly_map = anomaly_map.squeeze(axis=0)
    anomaly_map = img_process(anomaly_map)
    
    
    
    output_img = opt_img.squeeze(axis=0).to('cpu').detach().numpy()
    output_img = output_img_process(output_img.copy())
    
    
    
    
    
    
    
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0, 'hspace': 0})
    
    axs[0].imshow(content_img_copy, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('content_img')
    axs[1].imshow(style_img_copy, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('style_img')
    
    axs[2].imshow(output_img, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('anomaly_score' + str(anomaly_score))
    plt.grid(False)
    
    plt.show()
    
    
    fig.savefig(os.path.join(dump_path, 'anomaly_map.png'), bbox_inches='tight')
    
    
    
    
     
    
    
    
    
    
    
    
    
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    if img_id == num_of_iterations-1 or (saving_freq > 0 and img_id % saving_freq == 0):
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        
        dump_img = cv.cvtColor(dump_img, cv.COLOR_BGR2GRAY)
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img)

    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')






def img_process(img):
    img = np.moveaxis(img, 0, 2)  
    
    
    img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    img = img / 255.0
    
    
    
    
    
    img = np.clip(img, 0, 1).astype(np.float32)
    
    
    
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    
    
    
    
    
    return img










    

    



def output_img_process(img):
    out_img = np.moveaxis(img, 0, 2)  
    
    dump_img = np.copy(out_img)
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    dump_img = dump_img / 255.0
    
    
    dump_img = cv.cvtColor(dump_img, cv.COLOR_BGR2GRAY)
    
    
    
    
    
    return dump_img
    

def get_anomaly(x1, x2):
        
        
        anomaly_map = np.abs(x1.cpu().detach().numpy() - x2.cpu().detach().numpy()) 
        anomaly_score = np.mean(anomaly_map, keepdims=True)
         
        return anomaly_map, anomaly_score


def prepare_model(model, device):
    
    experimental = False
    if model == 'vgg16':
        if experimental:
            
            model = Vgg16Experimental(requires_grad=False, show_progress=True)
        else:
            model = Vgg16(requires_grad=False, show_progress=True)
    elif model == 'vgg19':
        model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise ValueError(f'{model} not supported.')

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
