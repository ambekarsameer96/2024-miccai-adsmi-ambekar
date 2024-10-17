import random
import os
import cv2
import numpy as np
import torch
from PIL import Image
import argparse
import gc
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as mods
import torchvision.transforms as transforms




from latent_ir.scripts.models.mlp_batch_norm import *
from latent_ir.scripts.models.utils import * 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wandb 

from torchmetrics.image import TotalVariation














































cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)



def img_load(path):
    img = cv2.imread(path)[::,::,::-1] 
    return img

def toPIL(img):
    img_type = str(type(img))
    if 'numpy' in img_type:
        img = Image.fromarray(img)
    elif 'torch' in img_type:
        img = transforms.ToPILImage()(img).convert("RGB")
    return img










def get_input_optimizer(input_img):
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
    return optimizer

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
      
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        
        return (img - self.mean) / self.std



















content_layers_default = ['relu_4']
style_layers_default = ['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    
    
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0  
    j = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            j+=1
            name = 'relu_{}'.format(j)
       
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        if isinstance(layer, nn.MaxPool2d):
            model.add_module(name, nn.AvgPool2d(2))
        else:
            model.add_module(name, layer)

        if name in content_layers:
            
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses










def total_variation_loss(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def mlp_forward(config_dict, network, coords, z):
    
    img_size = int(config_dict['mlp_img_size'])
    output = network(coords, z.repeat(img_size**2, 1))
    output = output.permute(1,0)
    output = output.view(-1,img_size,img_size)
    output = torch.unsqueeze( output, dim=0 )
    return output.type(torch.cuda.FloatTensor).to(device)

def reweighting(alpha, k=1):

    return (-1* (1+(alpha-1))**k) * ( torch.log(-alpha+1) )

def get_smooth(I, direction):
        
        weights = torch.tensor([[0., 0.],
                                [-1., 1.]]
                                ).cuda()
        weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
        weights_y = torch.transpose(weights_x, 0, 1)
        if direction == 'x':
            weights = weights_x
        elif direction == 'y':
            weights = weights_y

        F = torch.nn.functional
        output = torch.abs(F.conv2d(I, weights, stride=1, padding=1))  
        return output

def get_gradients_loss(I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    I_gray = torch.mean(I, dim=1, keepdim=True)
    gradients_I_x = get_smooth(I_gray,'x')
    gradients_I_y = get_smooth(I_gray,'y')

    return torch.mean(gradients_I_x * torch.exp(-10 * avg(R_gray, 'x')) + gradients_I_y * torch.exp(-10 * avg(R_gray, 'y')))


def avg(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

def run_style_transfer_inr(cnn,network,optimizer, normalization_mean, normalization_std,
                       content_img, style_img, coords ,latent_c,latent_s, num_steps,
                       style_weight, content_weight, config_dict):
    print('Building the style transfer model..')
    
    
    
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0, last_epoch=-1)

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    model.requires_grad_(False) 

    
    start = time.time()

    
   
    z_c = latent_c
    z_s = latent_s

    
    for i in range(config_dict['mlp_iter']- config_dict['mlp_start_iter']):
        total_step = i + config_dict['mlp_start_iter']
        network.train()

        alpha_1 = np.random.rand(1) 
        alpha_1 = torch.cuda.FloatTensor(alpha_1).to(device).detach()
      
        style_score_1   = torch.tensor(0., device=device)
        content_score_1 = torch.tensor(0., device=device)
   
        z_1 = alpha_1*z_c + (1-alpha_1)*z_s 
        

        input_img_1 = mlp_forward(config_dict, network, coords, z_1)
        input_img_1.requires_grad_(True)

        
        model(input_img_1) 
        for i, sl in enumerate(style_losses):
            style_score_1 += sl.loss
        for j, cl in enumerate(content_losses):
            content_score_1 += cl.loss
            
        

        weighted_style_score_1 = style_weight*(style_score_1)
        weighted_content_score_1 = content_weight*(content_score_1)
        reweighted_content_score_1 = reweighting(alpha_1,config_dict['mlp_kappa'])*weighted_content_score_1
        reweighted_style_score_1 = reweighting( (1-alpha_1), config_dict['mlp_kappa'])*weighted_style_score_1
        
        if(config_dict['mlp_consistency_weight']>0):
            content_img_noise = 0.95 * content_img + 1 * 0.05 * torch.randn_like(content_img)
            content_img_noise_output = model(content_img_noise)
            gen_img = model(input_img_1)
            mse_loss = torch.nn.MSELoss()
            consistency_loss = mse_loss(content_img_noise_output, gen_img)
            reweightwed_consistency_loss = config_dict['mlp_consistency_weight']*consistency_loss
            
        else:
            reweightwed_consistency_loss = 0
        
            
            

        
        
        
        
        metric = TotalVariation()
        metric = metric.to(device)
        tv_loss = metric(input_img_1)
        reweighted_tv_loss = config_dict['mlp_tv_weight'] * tv_loss
        
        
        
    
        smooth_loss = get_gradients_loss(input_img_1.detach().clone(), style_img.detach().clone())
        reweighted_smooth_loss = config_dict['mlp_smooth_weight']*smooth_loss
        
    
        
        loss_1 = reweighted_content_score_1 + reweighted_style_score_1  + reweightwed_consistency_loss + reweighted_smooth_loss
        loss = loss_1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()

        
        
        
        
        
        
        
        
        img_name = config_dict['mlp_image_name']
        
        
        
        wandb.log({'loss_image/' + img_name+"_loss": loss.item(), 'loss_image/' + img_name+"_content_loss": (content_weight*content_score_1).item(), 'loss_image/' + img_name+"_style_loss": (style_weight*style_score_1).item(),  'loss_image/' + img_name+"_smooth_loss": smooth_loss.item(), 'loss_image/' + img_name+"consistency_loss": consistency_loss.item(),})

        
        dec_result_1 = input_img_1.clone().detach()
    

        
        
        
        
        

        
        
          
        

        
        
        
        
        
        
        
        
            
        
            
            
            
        
          
            
            
            
            
            

            
            
            

    
    total_time = time.time() - start
    
    total_time = total_time/60
    print(f"{config_dict['mlp_type']} time :", total_time)
    wandb.log({'mlp_time': total_time})
    result_1 = ( np.clip(( dec_result_1[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
            
    
    img_1 = dec_result_1.detach().clone()
    
    
    
    img_size = int(config_dict['mlp_img_size'])
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    outputs= []
    results_np_List = []
    for alpha in alphas:
        z = (alpha * z_c + (1-alpha) * z_s).to(device)

        output = network(coords, z.repeat(img_size*img_size, 1)) 
        output = output.permute(1,0)
        output = output.view(-1,img_size,img_size)
        output = torch.unsqueeze( output, dim=0 )

        output = output.type(torch.cuda.FloatTensor).to(device)        
        result = output.clone().detach()
        outputs.append(result)
        
        result_np = ( np.clip(( result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
        results_np_List.append(result_np)
    
        
        
        
        
        
    
    
    
    
    
    return outputs, results_np_List, img_1
       
    


















        







  




   




















    







      






def configure_model_gnorm(model):
    
    
    model.train()
    
    model.requires_grad_(False)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model



def collect_params(network):
    
    params = []
    names = []
    for nm, m in network.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
    
    
def run_style_mlp(content, style, config_dict,epoch):
    
    
    
    
    random_seed = 1006
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  
    
    
    np.random.seed(random_seed)
    random.seed(random_seed)

    
    torch.cuda.empty_cache() 

    
        
    
    
    
    
    
    
    
    
  
    
     
    
    
   
    transform_list = []
    
    img_size = (config_dict['mlp_img_size'], config_dict['mlp_img_size'])
    
    
    
    
    

    
    
    
    
    
    content = content.type(torch.cuda.FloatTensor).to(device)
    style = style.type(torch.cuda.FloatTensor).to(device)


    
    
    
    
    
    
        
        
    cnn = mods.vgg19(pretrained=True).features.to(device).eval()

    network, optimizer, coords = get_network(config_dict,config_dict['mlp_img_size'])
    
    
    
    
    
    latent_size = config_dict['mlp_latent_size']
    z_c = torch.normal( 0, 1. , size=(latent_size,), device=device)
    z_s = torch.normal( 0, 1. , size=(latent_size,), device=device)

    
    
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
            
            
    
    outputs, result_np, img_1= run_style_transfer_inr(cnn,network,optimizer, cnn_normalization_mean,cnn_normalization_std, content,style,coords,z_c,z_s,num_steps=config_dict['mlp_iter'],style_weight=config_dict['mlp_style_wt'], content_weight=config_dict['mlp_content_wt'], config_dict=config_dict)
    
    
    
    
        
    
    del cnn
    
    
    
    return outputs, result_np, img_1
    