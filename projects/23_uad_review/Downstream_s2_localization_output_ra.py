import logging
from torch.nn import L1Loss
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns

import torch 
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score, roc_curve
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter

import lpips
from model_zoo.vgg import VGGEncoder

from optim.metrics import *
from core.DownstreamEvaluator import DownstreamEvaluator






import projects.tta_neural as tta_neural

import wandb 


class PDownstreamEvaluator(DownstreamEvaluator):
    
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)
        self.global_=False

    def start_task(self, global_model, params_dict):
        
        
        
        
        
        
        
        
        
        
        
        th = 0.08
        
        print('RUn above first ')
        print('Run after threshold')
        print('Replace with new value')
        print('Using threshold value of', th)
        
        self.object_localization(global_model, th, params_dict)
        
        
        

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    

    
    
    
    
    

    
    
    
    

    
    
    
    

    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    

    
    
    
    

    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    

    
    

    

    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    

    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

    

    def thresholding(self, global_model):
        
        logging.info("################ Threshold Search #################")
        self.model.load_state_dict(global_model)
        self.model.eval()
        ths = np.linspace(0, 1, endpoint=False, num=1000)
        fprs = dict()
        for th_ in ths:
            fprs[th_] = []
        im_scale = 128 * 128
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                for i in range(len(x)):
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    _,  x_res_i, _ = self.model.get_anomaly_ir(x_rec_i, x_i)
                    

                    
                    
                    
                    
                    for th_ in ths:
                        fpr = (np.count_nonzero(x_res_i > th_) * 100) / im_scale
                        fprs[th_].append(fpr)
        mean_fprs = []
        for th_ in ths:
            mean_fprs.append(np.mean(fprs[th_]))
        mean_fprs = np.asarray(mean_fprs)
        sorted_idxs = np.argsort(mean_fprs)
        th_1, th_2, best_th = 0, 0, 0
        fpr_1, fpr_2, best_fpr = 0, 0, 0
        for sort_idx in sorted_idxs:
            th_ = ths[sort_idx]
            fpr_ = mean_fprs[sort_idx]
            if fpr_ <= 1:
                th_1 = th_
                fpr_1 = fpr_
            if fpr_ <= 2:
                th_2 = th_
                fpr_2 = fpr_
            if fpr_ <= 5:
                best_th = th_
                best_fpr = fpr_
            else:
                break
        print(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        print('Best th: ', best_th)
        return best_th
    

    def object_localization(self, global_model,th=0, config_dict=None):
        
        logging.info("################ Object Localzation TEST #################" + str(th))
        lpips_alex = lpips.LPIPS(net='alex')  
        
        
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
                'TP': [],
                'FP': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            tps, fns, fps = 0, 0, []
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                
                x = data0.to(self.device)
                
                masks = data[1][:, 0, :, :].view(nr_batches, 1, width, height).to(self.device)\
                    if masks_bool else None
                neg_masks = data[1][:, 1, :, :].view(nr_batches, 1, width, height).to(self.device)
                neg_masks[neg_masks>0.5] = 1
                neg_masks[neg_masks<1] = 0
                
                anomaly_map_theta, _, x_rec_dict_org = self.model.get_anomaly(x, x)

                x_rec_org = x_rec_dict_org['x_rec'] if 'x_rec' in x_rec_dict_org.keys() else torch.zeros_like(x)
                x_rec_org = torch.clamp(x_rec_org, 0, 1)
                x_rec_org_copy = x_rec_org.detach().clone()
                
                
                
                cd = config_dict['configration_dict']
                cd['mlp_image_name'] = str(idx) + '.png'
                
                
                
                
                
                
                
                
                
                x_rec = x_rec_org.detach().clone()
                
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    x_res_i, _, _ = self.model.get_anomaly_ir(x,x_rec)
                    
                    
                    saliency_i = None
                    
                    
                    
                    

                    mask_ = masks[i][0].cpu().detach().numpy() if masks_bool else None
                    neg_mask_ = neg_masks[i][0].cpu().detach().numpy() if masks_bool else None
                    bboxes = cv2.cvtColor(neg_mask_*255, cv2.COLOR_GRAY2RGB)
                    
                    cnts_gt = cv2.findContours((mask_*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []
                    for c_gt in cnts_gt:
                        x, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([x, y, x+w, y+h])
                        cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    
                    
                    
                    
                    

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)
                    

                    x_combo = copy.deepcopy(x_res_i[0])
                    x_combo[x_combo < th] = 0

                    x_pos = x_combo * mask_
                    x_neg = x_combo * neg_mask_
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)
                    
                    x_rec_org_copy = x_rec_org_copy[i][0].cpu().detach().numpy()

                    amount_anomaly = np.count_nonzero(x_pos)
                    amount_mask = np.count_nonzero(mask_)

                    tp = 1 if amount_anomaly > 0.1 * amount_mask else 0
                    tps += tp
                    fn = 1 if tp == 0 else 0
                    fns += fn

                    fp = int(res_healthy / max(res_anomaly,1)) 
                    fps.append(fp)
                    precision = tp / max((tp+fp), 1)
                    test_metrics['TP'].append(tp)
                    test_metrics['FP'].append(fp)
                    test_metrics['Precision'].append(precision)
                    test_metrics['Recall'].append(tp)
                    test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

                    ious = [res_anomaly, res_healthy]

                    if (idx % 10001) == 0: 
                        elements = [x_, x_rec_org_copy, x_rec_, x_res_i, x_combo]
                        v_maxs = [1, 1, 1, 0.5, 0.5]
                        titles = ['Input', 'Source model', 'Rec', str(ious), '5%FPR']
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        

                        if masks_bool:
                            
                            
                            
                            elements.append(bboxes.astype(np.int64))
                            elements.append(x_pos)
                            elements.append(x_neg)
                            elements.append(x_combo)
                            
                            
                            v_maxs.append(1)
                            v_maxs.append(np.max(x_res_i))
                            v_maxs.append(np.max(x_res_i))
                            v_maxs.append(np.max(x_combo))
                            
                            
                            titles.append('GT')
                            titles.append(str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp))
                            titles.append(str(np.round(res_healthy, 2)) + ', FP: ' + str(fp))
                            titles.append('Combo')
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'plasma'
                            
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

                            wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                wandb.Image(diffp, caption="Sample_" + str(count))]})


            
            
            
            
            
            
            
            
            

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                if metric == 'TP':
                    logging.info(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
                if metric == 'FP':
                    logging.info(f'FP: {np.sum(test_metrics[metric])} missed')
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

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    

    
    

    

    
    
    
    
    

    
    
    

    
    
    

    
    
    
    
    
    
    
    
    
    

    

