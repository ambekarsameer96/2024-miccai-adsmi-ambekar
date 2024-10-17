## import RA 
## import PHI 

from model_zoo import ra

import torch.nn as nn





from phi_model.model_defn import *


import segmentation_models_pytorch as smp

import torch 

from torch.optim import Adam

from scipy.ndimage import gaussian_filter

import lpips 



model_name = 'vgg19'
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')



class PSI(nn.Module):
    def __init__(self, cdim=1, zdim=128, channels=(64, 128, 256, 512, 512), image_size=128, conditional=False,
                 cond_dim=10):
        super(PSI, self).__init__()

        self.zdim = zdim
        self.conditional = conditional
        self.cond_dim = cond_dim
        

        self.RA = ra.RA(cdim, zdim=zdim, channels=channels, image_size=image_size, conditional=False)
        

        
        
        self.PHI = smp.Unet(
            encoder_name="efficientnet-b4",        
            encoder_weights="imagenet",     
            in_channels=2,                  
            classes=1, 
            
            activation='sigmoid'
            
        )

        self.PHI =  self.PHI.to(device)
        
        self.VGG, self.content_feature_maps_index_name, self.style_feature_maps_indices_names = prepare_model(model_name, device)
        
        
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(device)



    def forward(self, x_s1, x_s2):
        x_s1_, _ = self.RA(x_s1)
        
        x_s2_re, _ = self.RA(x_s2)
        
        
        
        
        x_ = torch.concat([x_s2, x_s1_], dim=1)
        
        
        
        
        
        x_s1_phi = self.PHI(x_)
        
        
        
        
        
        
        
        return x_s1_, x_s1_phi, x_s2_re, {'useless_dict': 0}
    
    
    
    
    
    
    
         
    
    
    
    def get_anomaly(self, x1, x2):
        
        x_s1_, x_s1_phi, x_s2_re, _ = self.forward(x1, x2)
        
        anomaly_map = np.abs(x1.cpu().detach().numpy() - x_s1_.cpu().detach().numpy()) 
        anomaly_score = np.mean(anomaly_map, keepdims=True)
         
        return anomaly_map, anomaly_score, {'x_rec': x_s1_}  
    
    def get_anomaly_neural(self, x1, x2_neural):
        
        
        
        x2_neural_copy  = x2_neural.clone()
        x2_neural_copy = x2_neural_copy.to(device)
        x2_neural_copy = x2_neural_copy/255.0
        
        x2_neural_copy = torch.clamp(x2_neural_copy, 0, 1)
        
        anomaly_map = np.abs(x1.cpu().detach().numpy() - x2_neural_copy.cpu().detach().numpy()) 
        anomaly_score = np.mean(anomaly_map, keepdims=True)
         
        return anomaly_map, anomaly_score, {'x_rec': x2_neural}  
    
    def get_anomaly_ir(self, x, x_rec):
        
        anomaly_maps, anomaly_scores = self.compute_anomaly(x, x_rec)

        return anomaly_maps, anomaly_scores, {'x_rec': x_rec}

    def compute_anomaly(self, x, x_rec):
        anomaly_maps = []
        for i in range(len(x)):
            x_res, saliency = self.compute_residual_neural(x[i][0], x_rec[i][0])
            anomaly_maps.append(x_res*saliency)
        anomaly_maps = np.asarray(anomaly_maps)
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        return anomaly_maps, anomaly_scores

    
    def get_anomaly_ir_backup(self, x1, x2_ir):
        
        
        
        x2_neural_copy  = x2_ir.clone()
        x2_neural_copy = x2_neural_copy.to(device)
        
        
        x2_neural_copy = torch.clamp(x2_neural_copy, 0, 1)
        
        anomaly_map = np.abs(x1.cpu().detach().numpy() - x2_neural_copy.cpu().detach().numpy()) 
        anomaly_score = np.mean(anomaly_map, keepdims=True)
         
        return anomaly_map, anomaly_score, {'x_rec': x2_neural_copy}
    
    
    
    
    
    

    
    
    ## NEW METRICS 

    
    
    
    
    
    
    
    

    def compute_residual_neural(self, x, x_rec):
        x = torch.clamp(x,0,1)
        x_rec = torch.clamp(x_rec, 0, 1)
        saliency = self.get_saliency_neural(x, x_rec)
        x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())
        return x_res, saliency
    
    def lpips_loss_neural(self, ph_img, anomaly_img, mode=0):
        def ano_cam(ph_img_, anomaly_img_):
            
            loss_lpips = self.l_pips_sq(anomaly_img_, ph_img_, normalize=True, retPerLayer=False)
            return loss_lpips.cpu().detach().numpy()

        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        ano_map = ano_cam(ph_img, anomaly_img)
        return ano_map
    
    

    def get_saliency_neural(self, x, x_rec):
        saliency = self.lpips_loss_neural(x_rec, x)
        saliency = gaussian_filter(saliency, sigma=2)
        return saliency
    
    
    
    
        
    


    
















    
    
    
    
    
        
         