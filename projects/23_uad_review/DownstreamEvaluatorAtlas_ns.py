import logging
import io

import matplotlib
import cv2 as cv 

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns


from torch.nn import L1Loss
from torch.cuda.amp import autocast
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch.nn.functional as F


from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter

from PIL import Image
import cv2
import numpy as np
import torchvision

import lpips

from dl_utils import *
from optim.metrics import *

from core.DownstreamEvaluator import DownstreamEvaluator
import os
import copy
from model_zoo.vgg import VGGEncoder


from projects.neural_style import neural_style_transfer

import torchshow 


class PDownstreamEvaluator(DownstreamEvaluator):
    

    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True,
                                     lpips=True).to(self.device)
        
        
        

        

        
        

        self.global_ = True

    def start_task(self, global_model):
        
        self.pathology_localization(global_model, 1, 71, True)
        self.pathology_localization(global_model, 71, 570, True)
        self.pathology_localization(global_model, 570, 10000, True)

    def _log_visualization(self, to_visualize, i, count):
        
        diffp, axarr = plt.subplots(1, len(to_visualize), gridspec_kw={'wspace': 0, 'hspace': 0},
                                    figsize=(len(to_visualize) * 4, 4))
        for idx, dict in enumerate(to_visualize):
            if 'title' in dict:
                axarr[idx].set_title(dict['title'])
            axarr[idx].axis('off')
            tensor = dict['tensor'][i].cpu().detach().numpy().squeeze() if isinstance(dict['tensor'], torch.Tensor) else \
            dict['tensor'][i].squeeze()
            axarr[idx].imshow(tensor, cmap=dict.get('cmap', 'gray'), vmin=dict.get('vmin', 0), vmax=dict.get('vmax', 1))
        diffp.set_size_inches(len(to_visualize) * 4, 4)

        wandb.log({f'Anomaly_masks/Example_Atlas_{count}': [wandb.Image(diffp, caption="Atlas_" + str(count))]})


    def find_mask_size_thresholds(self, dataset):
        
        mask_sizes = []
        for _, data in enumerate(dataset):
            if 'dict' in str(type(data)) and 'images' in data.keys():
                data0 = data['images']
            else:
                data0 = data[0]
            x = data0.to(self.device)
            masks = data[1].to(self.device)
            masks[masks > 0] = 1

            for i in range(len(x)):
                if torch.sum(masks[i][0]) > 1:
                    mask_sizes.append(torch.sum(masks[i][0]).item())

        unique_mask_sizes = np.unique(mask_sizes)
        print(type(unique_mask_sizes))
        lower_tail_threshold = np.percentile(unique_mask_sizes, 25)
        upper_tail_threshold = np.percentile(unique_mask_sizes, 75)

        _ = plt.figure()
        
        plt.hist(mask_sizes, bins=100)
        plt.xlabel('Mask Sizes')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mask Sizes')

        plt.axvline(lower_tail_threshold, color='r', linestyle='--', label=f'25th Percentile: {lower_tail_threshold}')
        plt.axvline(upper_tail_threshold, color='g', linestyle='--', label=f'75th Percentile: {upper_tail_threshold}')
        print(lower_tail_threshold, upper_tail_threshold)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({"Anomaly/Mask sizes1": [wandb.Image(Image.open(buf), caption="Mask Sizes")]})

        plt.clf()

    def pathology_localization(self, global_model, threshold_low, threshold_high, perc_flag=False):
        
        logging.info(f"################ Stroke Anomaly Detection {threshold_low} - {threshold_high} #################")
        lpips_alex = lpips.LPIPS(net='alex')

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
        }
        pred_dict = dict()
        
        optimization_config = dict()
                
        
        
        
        
        
            
        
        
            
        
        
            
        
        
        
        
        
        
        
        optimization_config = {'height': 128, 'content_weight': 10000000.0, 'style_weight': 100000.0, 'tv_weight': 100.0, 'adam_lr': 0.1, 'optimizer': 'adam', 'model': 'vgg19', 'saving_freq': 2000, 'output_iter': 10000, 'consistency_weight': 1}
        
        
        print('optimization_config is ', optimization_config)
        
        
        
        
        wandb.config.update(optimization_config)

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
            }
            global_counter = 0
            threshold_masks = []
            anomalous_pred = []
            healthy_pred = []

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):

                
                
                
                if idx >10 :
                    break

                
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1
                
                
                print('Main Loop idx is ', idx)
                
                

                
                anomaly_map_theta, _, x_rec_dict_org = self.model.get_anomaly(x, x)

                x_rec_org = x_rec_dict_org['x_rec'] if 'x_rec' in x_rec_dict_org.keys() else torch.zeros_like(x)
                x_rec_org = torch.clamp(x_rec_org, 0, 1)
                
                
                
                
                style_img = x.detach().clone()
                
                style_img = style_img.repeat(1, 3, 1, 1)
                
                
                style_img = style_img.squeeze()
                
                
                
                
                
                
                
                
                
                
                context_img = x_rec_org.detach().clone()
                context_img = context_img.repeat(1, 3, 1, 1)
                context_img = context_img.squeeze()
                
                
                
                
                
                
                
                
                
                
                
                
                context_img = neural_image_pre_process(context_img)
                
                style_img = neural_image_pre_process(style_img)
                
                
                
                
                
                
                
                
                
                iter = idx 
                
                
                
                
                
                
                
                
                
                
                
                
                outputs = neural_style_transfer(context_img.detach().clone(), style_img.detach().clone(), optimization_config)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly_neural(x, torchvision.transforms.functional.rgb_to_grayscale(outputs[0]))
                x_rec_dict['x_rec'] = neural_image_post_process(x_rec_dict['x_rec'])
                
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                
                
                
                
                
                
                
                
                
                
                
                
                wandb.log({'content': [wandb.Image(context_img, caption="content" + str(idx))]})
                wandb.log({'style': [wandb.Image(style_img, caption="style" + str(idx))]})
                
                
                wandb.log({'x': [wandb.Image(x, caption="x" + str(idx))]})
                
                wandb.log({'x_rec_org': [wandb.Image(x_rec_org, caption="x_rec_org" + str(idx))]})
                
                wandb.log({'anomaly_map_theta': [wandb.Image(anomaly_map_theta, caption="anomaly_map_theta" + str(idx))]})
                
                wandb.log({'anomaly_map_ns': [wandb.Image(anomaly_map, caption="anomaly_map" + str(idx))]})
                
                
                wandb.log({'x_rec': [wandb.Image(x_rec, caption="x_rec" + str(idx))]})
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                 
                
                

                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec_theta', 'tensor': x_rec_org},
                    
                    {'title': 'x_rec_ns', 'tensor': x_rec},
                    
                    {'title': f'Anomaly  map theta {anomaly_map_theta.max():.3f}', 'tensor': anomaly_map_theta, 'cmap': 'plasma',
                     'vmax': anomaly_map_theta.max()},
                    
                    {'title': f'Anomaly  map ns {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': anomaly_map.max()},
                    {'title': 'gt', 'tensor': masks, 'cmap': 'plasma'}
                ]

                if 'mask' in x_rec_dict.keys():
                    masked_input = x_rec_dict['mask'] + x
                    masked_input[masked_input>1]=1

                    to_visualize.append({'title': 'Rec Orig', 'tensor': x_rec_dict['x_rec_orig'], 'cmap': 'gray'})
                    to_visualize.append({'title': 'Res Orig', 'tensor': x_rec_dict['x_res'], 'cmap': 'plasma',
                                        'vmax': x_rec_dict['x_res'].max()})
                    to_visualize.append({'title': 'Mask', 'tensor': masked_input, 'cmap': 'gray'})
                
                
                
                
                
                for i in range(len(x)):
                    if torch.sum(masks[i][0]) > threshold_low and torch.sum(
                            masks[i][0]) <= threshold_high:  
                    
                    
                    
                        count = str(idx * len(x) + i)
                        
                        if int(count) in [100, 105, 112, 121, 186, 189, 210, 214, 345, 382, 424, 425, 435, 434, 441,
                                          462, 464, 472, 478, 504]:
                            print("skipping ", count)
                            continue

                        
                        if int(count) % 12 == 0 or int(count) in [0, 66, 325, 352, 545, 548, 231, 609, 616, 11, 254,
                                                                  539, 165, 545, 550, 92, 616, 628, 630, 636, 651]:
                            
                            pass
                        
                        self._log_visualization(to_visualize, i, count)

                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        res_2_i_np = anomaly_map[i][0]
                        anomalous_pred.append(anomaly_score[i][0])

                        pred_.append(res_2_i_np)
                        label_.append(masks[i][0].cpu().detach().numpy())

                        
                        loss_mae = torch.mean(torch.abs(rec_2_i - x_i))
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), rec_2_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(rec_2_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

                    elif torch.sum(
                            masks[i][0]) <= 1:  
                        res_2_i_np_healthy = anomaly_map[i][0]
                        healthy_pred.append(anomaly_score[i][0])
                        
                        
                        
                        


                    
                    
                    
                    
                    
                    

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{}: {} mean: {} +/- {}'.format(dataset_key, metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        for dataset_key in self.test_data_dict.keys():
            
            pred_ood, label_ood = pred_dict[dataset_key]
            predictions = np.asarray(pred_ood)
            labels = np.asarray(label_ood)
            predictions_all = np.reshape(np.asarray(predictions), (len(predictions), -1))  
            labels_all = np.reshape(np.asarray(labels), (len(labels), -1))  
            print(f'Nr of preditions: {predictions_all.shape}')
            print(
                f'Predictions go from {np.min(predictions_all)} to {np.max(predictions_all)} with mean: {np.mean(predictions_all)}')
            print(f'Labels go from {np.min(labels_all)} to {np.max(labels_all)} with mean: {np.mean(labels_all)}')
            print('Shapes {} {} '.format(labels.shape, predictions.shape))

            
            dice_scores = []

            auprc_, _, _, _ = compute_auprc(predictions_all, labels_all)
            logging.info(f'Global AUPRC score: {auprc_}')
            wandb.log({f'Metrics/{threshold_low}_Global_AUPRC_{dataset_key}': auprc_})

            
            ths = np.linspace(0, 1, 101)
            for dice_threshold in ths:
                dice = compute_dice(copy.deepcopy(predictions_all), copy.deepcopy(labels_all), dice_threshold)
                dice_scores.append(dice)
            highest_score_index = np.argmax(dice_scores)
            highest_score = dice_scores[highest_score_index]

            logging.info(f'Global highest DICE: {highest_score}')
            wandb.log({f'Metrics/{threshold_low}_Global_highest_DICE': highest_score})

        
        logging.info('Writing plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
            wandb.log({f'Metrics/{threshold_low}_{self.name}_{metric}': fig_bp})
            
            
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]
def neural_image_pre_process(img):
    
    
    img = img * 255
    
    
    img = img.unsqueeze(0)
    return img.to(device)












    




    
    









device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_image_post_process(img):
    
    
    
    
    
    out_img = img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)  

    dump_img = np.copy(out_img)
    
    dump_img = np.repeat(dump_img, 3, axis=2)
    
    dump_img = np.clip(dump_img, 0, 255)
    
    dump_img = cv.cvtColor(dump_img, cv.COLOR_BGR2GRAY)
    
    dump_img = dump_img[np.newaxis, :, :]
    dump_img = (dump_img / 255.0).astype(np.float32)
    
    img_new = torch.from_numpy(dump_img).to(device)
    
    img_new = img_new.unsqueeze(0)
    img_new = img_new.to(device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return img_new




    
    
    







    








    









    

    
    





    
    