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


import torchvision.transforms as transforms
from transforms.preprocessing import *

from torch.nn import L1Loss
from torch.cuda.amp import autocast
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch.nn.functional as F


from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import roc_auc_score, roc_curve

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

from latent_ir.scripts import optimization as optimization_ir

import torchvision.transforms as transforms
from transforms.preprocessing import *

import skimage.measure

class PDownstreamEvaluator(DownstreamEvaluator):
    

    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True,
                                     lpips=True).to(self.device)
        
        
        

        

        
        

        self.global_ = True

    def start_task(self, global_model, params_dict=None):
        
        
        
        
        
        
        
        
        print('=====Calling large sizes for comutation=====')
        
        self.pathology_localization(global_model, 570, 10000, True,params_dict)
        
        
        
        
        

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

    def pathology_localization(self, global_model, threshold_low, threshold_high, perc_flag=False, params_dict=None):
        
        logging.info(f"################ Stroke Anomaly Detection {threshold_low} - {threshold_high} #################")
        lpips_alex = lpips.LPIPS(net='alex')

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
            'FP': [],
            'FN': [],
            'TP': [],
            'TN': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'Accuracy': [],
        }
        pred_dict = dict()
        
        wandb_folder_name = wandb_run_name = wandb.run.name

        optimization_config = dict()
        MODEL_DIR = os.path.join('./saved_outputs', wandb_run_name)
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
    
        
        fol1 = os.path.join(MODEL_DIR, 'content_images')
        if not os.path.exists(fol1):
            os.makedirs(fol1)
            
        fol2 = os.path.join(MODEL_DIR, 'style_images')
        if not os.path.exists(fol2):
            os.makedirs(fol2)
            
        fol3 = os.path.join(MODEL_DIR, 'x_rec_images')
        if not os.path.exists(fol3):
            os.makedirs(fol3)
            
        
        wandb.log({'content_dir': fol1})
        wandb.log({'style_dir': fol2})
        wandb.log({'x_rec_dir': fol3})
        print('====================== Saving images to directory ======================')
        print('Content directory is ', fol1)
        print('Style directory is ', fol2)
        print('X_rec directory is ', fol3)
        
        config_dict = params_dict['configration_dict']
        
        
        
        
            
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        print('==============================')
        print('Config dict is ', config_dict)
        print('==============================')

        
        
        
        
        
            
        
        
            
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        wandb.config.update(config_dict)

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dice_mask_list  = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
                'FP': [],
                'FN': [],
                'TP': [],
                'TN': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
                'Accuracy': [],
                
            }
            global_counter = 0
            threshold_masks = []
            anomalous_pred = []
            healthy_pred = []

            logging.info('DATASET: {}'.format(dataset_key))
            
            
            wandb.define_metric("tta_step")
            wandb.define_metric("content", step_metric = "tta_step")
            wandb.define_metric("style", step_metric = "tta_step")
            wandb.define_metric("x", step_metric = "tta_step")
            wandb.define_metric("x_rec_org", step_metric = "tta_step")
            wandb.define_metric("anomaly_map_theta", step_metric = "tta_step")
            wandb.define_metric("anomaly_map_ns", step_metric = "tta_step")
            wandb.define_metric("x_rec", step_metric = "tta_step")
            wandb.define_metric("output_x_res", step_metric = "tta_step")
            wandb.define_metric("output_saliency", step_metric = "tta_step")
            wandb.define_metric("output_ano_map", step_metric = "tta_step")
            
            count_img = 1 
            
            total_count_ds = len(dataset)
            wandb.log({'total_count_ds': total_count_ds})
            
            
            entropy_input_list = []
            entropy_output_list = []
            entropy_theta_list = []

            for idx, data in enumerate(dataset):

                
                
                
                

                
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1
                
                
                print('Main Loop idx is ', idx)
                
                
                if (torch.sum(masks[0][0]) <= threshold_low) or (torch.sum(masks[0][0]) > threshold_high):  
                    print('=======The skipped idx is *************', idx)
                    wandb.log({'skipped_idx': idx})
                    
                    continue
                else:
                    
                    count_img += 1
                    wandb.log({'count_img': count_img})
                    print('Count_img is ', count_img)
                    
                    
                
                
                

                
                
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(x, x)
                del anomaly_map, anomaly_score

                
                
                
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)
                
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly_ir(x, x_rec)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                iter = idx 
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                config_dict['mlp_image_name'] = str(idx) + '.png'
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                    
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                wandb.log({'tta_step': idx})
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                dice_mask_mode = True
                
                if dice_mask_mode:
                    folder_name = './data/ATLAS/masks/masks_all/'
                    
                    
                    RES = transforms.Resize((128, 128))
                    
                    img_path = os.path.join(folder_name, str(idx) + '.png')
                    transform_load = transforms.Compose([ReadImage(), To01()
                                            
                                            ,AddChannelIfNeeded()
                                            ,AssertChannelFirst(), RES
                                            ])
                    
                    
                    dice_mask = transform_load(img_path)
                    dice_mask = dice_mask.to(device)
                    dice_mask = dice_mask.unsqueeze(0)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                 
                
                
                
                
                
                
                sal_max = 0.6

                
                count_img += 1
                wandb.log({'tta_step': idx})
                wandb.log({'count_img': count_img})
                
                
                
                 
                
                
                
                
                
                val_sal = 0.6

                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec_theta', 'tensor': x_rec},
                    
                    {'title': 'x_rec_hist', 'tensor': x_rec},
                    
                    {'title': f'Anomaly  map theta {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': 0.999},
                    
                    {'title': f'Anomaly  map ns {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': 0.999},
                    {'title': 'gt', 'tensor': masks, 'cmap': 'plasma'},
                    {'title': f'Dice mask {dice_mask.max():.3f}', 'tensor': dice_mask, 'cmap': 'plasma',
                        'vmax': dice_mask.max()},
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
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
                            self._log_visualization(to_visualize, i, count)
                            
                        
                        self._log_visualization(to_visualize, i, count)

                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        res_2_i_np = anomaly_map[i][0]
                        anomalous_pred.append(anomaly_score[i][0])
                        
                        
                        dice_mask_i_np = dice_mask[i]
                        
                        
                        dice_mask_i_np = dice_mask_i_np.clone()
                        dice_mask_i_np[dice_mask_i_np > 0.5] = 1
                        dice_mask_i_np[dice_mask_i_np <= 0.5] = 0
                        

                        pred_.append(res_2_i_np)
                        label_.append(masks[i][0].cpu().detach().numpy())
                        dice_mask_list.append(dice_mask_i_np.cpu().detach().numpy())
                       
                        
                        loss_mae = torch.mean(torch.abs(rec_2_i - x_i))
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), rec_2_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(rec_2_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)
                        
                        fp = np.mean(((1- masks[i][0].cpu().detach().numpy()) * res_2_i_np))
                        
                        test_metrics['FP'].append(fp)
                        
                        fn = np.mean((masks[i][0].cpu().detach().numpy() * (1- res_2_i_np)))
                        
                        tp = np.mean((masks[i][0].cpu().detach().numpy() * res_2_i_np))
                        
                        tn = np.mean((1- masks[i][0].cpu().detach().numpy()) * (1- res_2_i_np))
                        
                        
                        test_metrics['FN'].append(fn)
                        test_metrics['TP'].append(tp)
                        test_metrics['TN'].append(tn)
                        
                        precision = tp / (tp + fp)
                        test_metrics['Precision'].append(precision)
                        
                        recall = tp / (tp + fn)
                        test_metrics['Recall'].append(recall)
                        
                        f1 = 2 * (precision * recall) / (precision + recall)
                        test_metrics['F1'].append(f1)
                        
                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        test_metrics['Accuracy'].append(accuracy)
                        
                    elif torch.sum(
                            masks[i][0]) <= 1:  
                        res_2_i_np_healthy = anomaly_map[i][0]
                        healthy_pred.append(anomaly_score[i][0])
                        
                        
                        
                        


                    
                    
                    
                    
                    
                    
                    
                     
                    
                    
                    
            
            pred_dict[dataset_key] = (pred_, label_, dice_mask_list)
            
            
            
            
            
            

            for metric in test_metrics:
                logging.info('{}: {} mean: {} +/- {}'.format(dataset_key, metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])
            
        
            
        for dataset_key in self.test_data_dict.keys():
            
            pred_ood, label_ood, dice_mask_img  = pred_dict[dataset_key]
            predictions = np.asarray(pred_ood)
            labels = np.asarray(label_ood)
            print('Shape before filtering is ', predictions.shape, labels.shape)
            
            predictions = np.squeeze(predictions)
            labels = np.asarray(label_ood)
            dice_mask_np = np.asarray(dice_mask_img)
            dice_mask_np = np.squeeze(dice_mask_np)
            
            
            fp_area = dice_mask_np - labels
            tp_area = labels 
            
            tp_new = np.mean(predictions * tp_area)
            fp_new = np.mean(predictions * fp_area)
            
            print('NEW TP is ', tp_new)
            print('NEW FP is ', fp_new)
            
            tp_new = tp_new * 1000
            fp_new = fp_new * 1000
            tp_new = round(tp_new, 6)
            fp_new = round(fp_new, 6)
            print('New rounded TP is ', tp_new)
            print('New rounded FP is ', fp_new)
            
            new_list , new_preds_2, new_labels_2= select_indices(predictions, labels, dice_mask_np)
            
            
            predictions = padding_func(new_preds_2)
            labels = padding_func(new_labels_2)
            
            
            
            
            
            print('Shape after filtering is ', predictions.shape, labels.shape)
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
            
            predictions_all_2 = predictions_all.flatten()
            labels_all_2 = labels_all.flatten()
            labels_all_2 = labels_all_2.astype(int)
            
            
            
            
            
            auroc =  roc_auc_score(labels_all_2, predictions_all_2)
            print('[ {} ]: AUROC: {}'.format(dataset_key, auroc))
            
            wandb.log({f'Metrics/{threshold_low}_Global_AUROC_{dataset_key}': auroc})
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            
            tp_scores = []
            fp_scores = []
            
            ths = np.linspace(0, 1, 101)
            for dice_threshold in ths:
                dice, tp, fp  = compute_dice_tp_fp(copy.deepcopy(predictions_all), copy.deepcopy(labels_all), dice_threshold)
                dice_scores.append(dice)
                tp_scores.append(tp)
                fp_scores.append(fp)
            highest_score_index = np.argmax(dice_scores)
            highest_score = dice_scores[highest_score_index]
            highest_tp = tp_scores[highest_score_index]
            highest_fp = fp_scores[highest_score_index]
            

            logging.info(f'Global highest DICE: {highest_score}')
            logging.info(f'Global highest TP: {highest_tp}')
            logging.info(f'Global highest FP: {highest_fp}')
            
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
            

    def healthy_recons(self, global_model, threshold_low, threshold_high, perc_flag=False, params_dict=None):
        
        logging.info(f"################ Stroke Anomaly Detection {threshold_low} - {threshold_high} #################")
        lpips_alex = lpips.LPIPS(net='alex')

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
            'FP' : [],
        }
        pred_dict = dict()
        
        
        optimization_config = dict()
        
        config_dict = params_dict['configration_dict']
        
        
        
        
            
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        print('==============================')
        print('Config dict is ', config_dict)
        print('==============================')

        
        
        
        
        
            
        
        
            
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        wandb.config.update(config_dict)

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
                'FP' : [],
            }
            global_counter = 0
            threshold_masks = []
            anomalous_pred = []
            healthy_pred = []

            logging.info('DATASET: {}'.format(dataset_key))
            
            
            wandb.define_metric("tta_step")
            wandb.define_metric("content", step_metric = "tta_step")
            wandb.define_metric("style", step_metric = "tta_step")
            wandb.define_metric("x", step_metric = "tta_step")
            wandb.define_metric("x_rec_org", step_metric = "tta_step")
            wandb.define_metric("anomaly_map_theta", step_metric = "tta_step")
            wandb.define_metric("anomaly_map_ns", step_metric = "tta_step")
            wandb.define_metric("x_rec", step_metric = "tta_step")
            wandb.define_metric("output_x_res", step_metric = "tta_step")
            wandb.define_metric("output_saliency", step_metric = "tta_step")
            wandb.define_metric("output_ano_map", step_metric = "tta_step")
            
            
            count_img = 0 

            for idx, data in enumerate(dataset):

                
                
                
                

                
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                if (len(data)>1):
                    masks = data[1].to(self.device)
                    masks[masks > 0] = 1
                else:
                    print('I am being called')
                    masks = torch.zeros_like(x)
                
                
                print('Main Loop idx is ', idx)
                print(torch.sum(masks[0][0]))
                
            
                if (torch.sum(masks[0][0]) <= threshold_low) or (torch.sum(masks[0][0]) > threshold_high):  
                    print('=======The skipped idx is *************', idx)
                    wandb.log({'skipped_idx': idx})
                    
                    continue
                else:
                    
                    count_img += 1
                    wandb.log({'count_img': count_img})
                    print('Count_img is ', count_img)
                    
                
                
                
                
                anomaly_map_theta, _, x_rec_dict_org = self.model.get_anomaly(x, x)

                x_rec_org = x_rec_dict_org['x_rec'] if 'x_rec' in x_rec_dict_org.keys() else torch.zeros_like(x)
                x_rec_org = torch.clamp(x_rec_org, 0, 1)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                anomaly_map_theta, _, x_rec_dict_org = self.model.get_anomaly(x, x)

                x_rec_org = x_rec_dict_org['x_rec'] if 'x_rec' in x_rec_dict_org.keys() else torch.zeros_like(x)
                x_rec_org = torch.clamp(x_rec_org, 0, 1)
                
                
                
                
                
                style_img = x.detach().clone()
                
                style_img = style_img.repeat(1, 3, 1, 1)
                
                
                
                
                
                
                
                
                
                
                
                
                
                context_img = x_rec_org.detach().clone()
                context_img = context_img.repeat(1, 3, 1, 1)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                iter = idx 
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                config_dict['mlp_image_name'] = str(idx) + '.png'
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                    
                    
                
                
                
                
                

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly_ir(x, output_next)
                
                
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                
                
                
                x_rec_org_res, x_rec_org_saliency = self.model.compute_residual_neural(x, x_rec_org)
                output_x_res, output_saliency = self.model.compute_residual_neural(x, output_next) 
                
                x_rec_org_ano_map = self.model.lpips_loss_neural(x,x_rec_org)
                
                output_ano_map = self.model.lpips_loss_neural(x, output_next)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                wandb.log({'tta_step': idx})
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                 
                
                
                
                
                
                
                sal_max = 0.6

                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec_theta', 'tensor': x_rec_org},
                    
                    {'title': 'x_rec_ns', 'tensor': x_rec},
                    
                    {'title': f'Anomaly  map theta {anomaly_map_theta.max():.3f}', 'tensor': anomaly_map_theta, 'cmap': 'plasma',
                     'vmax': anomaly_map_theta.max()},
                    
                    {'title': f'Anomaly  map ns {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': anomaly_map.max()},
                    {'title': 'gt', 'tensor': masks, 'cmap': 'plasma'},
                    
                    
                    
                    {'title': f'Residual map theta {x_rec_org_res.max():.3f}', 'tensor': x_rec_org_res, 'cmap': 'plasma',
                        'vmax': anomaly_map.max()},
                    
                    {'title': f'Residual map ns {output_x_res.max():.3f}', 'tensor': output_x_res, 'cmap': 'plasma',
                        'vmax': output_x_res.max()},
                    
                    
                    {'title': f'Saliency map theta {sal_max}', 'tensor': x_rec_org_saliency, 'cmap': 'plasma',
                        'vmax':sal_max},
                    
                    {'title': f'Saliency map ns {sal_max}', 'tensor': output_saliency, 'cmap': 'plasma',
                        'vmax': sal_max},
                    
                    {'title': f'Ano map theta {x_rec_org_ano_map.max():.3f}', 'tensor': x_rec_org_ano_map, 'cmap': 'plasma',
                        'vmax': x_rec_org_ano_map.max()},
                    
                    {'title': f'Ano map ns {output_ano_map.max():.3f}', 'tensor': output_ano_map, 'cmap': 'plasma',
                        'vmax': output_ano_map.max()}
                                ]

                
                
                

                
                
                
                
                
                
                
                
                
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
                            self._log_visualization(to_visualize, i, count)
                            
                        
                        self._log_visualization(to_visualize, i, count)

                        x_i = x[i][0]
                        

                        
                        

                        
                        

                        
                        
                        
                        
                        
                        
                        loss_mae  = torch.mean(torch.abs( x_rec[i][0].cpu().detach() - x_i.cpu().detach()))
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec[i][0].cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(x_rec[i][0].cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)
                        
                        

                    
                        


                    
                    
                    
                    
                    
                    

            

            for metric in test_metrics:
                logging.info('{}: {} mean: {} +/- {}'.format(dataset_key, metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        

        
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




    
    
    







    








    









    

    
    





def hist_pre_process(img_1):
    
    img_1 = img_1.squeeze(0)
    img_1 = transforms.Grayscale(num_output_channels=1)(img_1)
    
    img_1 = img_1.cpu().detach().numpy()
    
    img_1 = np.clip(img_1, 0, 1)
    
    img_1 = np.transpose(img_1, (1, 2, 0))
    
    
    
    img_1 = img_1.astype(np.float32)
    return img_1



def hist_post_process(img_2):
    
    img_2 = torch.from_numpy(img_2)
    img_2 = torch.tensor(img_2, dtype=torch.float32)
    
    img_2 = img_2.permute(2, 0, 1)
    
    
    
    
    
    
    
    
    img_2 = img_2.unsqueeze(0)
    
    img_2 = img_2.to(device)
    return img_2
    
    










def select_indices(predictions, labels, dice_mask_np):
    
    selected_values = []
    pred_list = []
    labels_list = []

    
    for i in range(dice_mask_np.shape[0]):
        
        mask_indices = np.where(dice_mask_np[i] == 1)
        

        
        pred_values = predictions[i][mask_indices]
        label_values = labels[i][mask_indices]
        
        pred_list.append(pred_values)
        labels_list.append(label_values)

        
        

    return selected_values, pred_list, labels_list



def padding_func(list):
    
    max_len = 0
    for i in list:
        if len(i) > max_len:
            max_len = len(i)
            
    
    
    new_list = []
    for i in list:
        padding = max_len - len(i)
        
        ele = np.pad(i, (0, padding), 'constant')
        new_list.append(ele)
        
    
    new_array = np.array(new_list)
    
    
    return new_array
        
        
        
