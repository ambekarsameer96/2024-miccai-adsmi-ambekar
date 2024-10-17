import logging
from torch.nn import L1Loss
import copy
import torch
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score, roc_curve


import lpips
from model_zoo.vgg import VGGEncoder

from optim.metrics import *
from core.DownstreamEvaluator import DownstreamEvaluator




class PDownstreamEvaluator(DownstreamEvaluator):
    
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.global_ = True

    def start_task(self, global_model):
        
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        th = 0.234 
        
        
        if self.global_:
            self.global_detection(global_model)
        else:
            self.object_localization(global_model, th)
        
        
        

    def _log_visualization(self, to_visualize, dataset_key, count):
        
        diffp, axarr = plt.subplots(1, len(to_visualize), gridspec_kw={'wspace': 0, 'hspace': 0},
                                    figsize=(len(to_visualize) * 4, 4))
        for idx, dict in enumerate(to_visualize):
            if 'title' in dict:
                axarr[idx].set_title(dict['title'])
            axarr[idx].axis('off')
            tensor = dict['tensor'].cpu().detach().numpy().squeeze() if isinstance(dict['tensor'], torch.Tensor) else \
            dict['tensor'].squeeze()
            axarr[idx].imshow(tensor, cmap=dict.get('cmap', 'gray'), vmin=dict.get('vmin', 0), vmax=dict.get('vmax', 1))
        diffp.set_size_inches(len(to_visualize) * 4, 4)

        wandb.log({f'Anomaly_masks/Example_FastMRI_{dataset_key}_{count}': [wandb.Image(diffp, caption="Atlas_" + str(
            count))]})

    def global_detection(self, global_model):
        
        logging.info("################ MANIFOLD LEARNING TEST #################")
        lpips_alex = lpips.LPIPS(net='alex')  

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': []
        }
        pred_dict = dict()
        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': []
            }
            logging.info('DATASET: {}'.format(dataset_key))
            
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(x)
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    anomaly_map_i = anomaly_map[i][0]
                    anomaly_score_i = anomaly_score[i][0]

                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    label = 0 if 'Normal' in dataset_key else 1
                    pred_.append(anomaly_score_i)
                    label_.append(label)

                    if int(count) % 3 == 0 or int(count) in [67]:
                        elements = [x_, x_rec_, anomaly_map_i]
                        v_maxs = [1, 1, np.max(anomaly_map_i)-0.0001]
                        titles = ['Input', 'Rec', 'Anomay Score: ' + str(anomaly_score_i)]

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

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        if self.compute_scores:
            normal_key = 'Normal'
            for key in pred_dict.keys():
                if 'Normal' in key:
                    normal_key = key
                    break
            pred_cxr, label_cxr = pred_dict[normal_key]
            for dataset_key in self.test_data_dict.keys():
                print(f'Running evaluation for {dataset_key}')
                if dataset_key == normal_key:
                    continue
                pred_ood, label_ood = pred_dict[dataset_key]
                predictions = np.asarray(pred_cxr + pred_ood)
                labels = np.asarray(label_cxr + label_ood)
                print('Negative Classes: {}'.format(len(np.argwhere(labels == 0))))
                print('Positive Classes: {}'.format(len(np.argwhere(labels == 1))))
                print('total Classes: {}'.format(len(labels)))
                print('Shapes {} {} '.format(labels.shape, predictions.shape))

                auprc = average_precision_score(labels, predictions)
                print('[ {} ]: AUPRC: {}'.format(dataset_key, auprc))
                auroc = roc_auc_score(labels, predictions)
                print('[ {} ]: AUROC: {}'.format(dataset_key, auroc))

                fpr, tpr, ths = roc_curve(labels, predictions)
                th_95 = np.squeeze(np.argwhere(tpr >= 0.95)[0])
                th_99 = np.squeeze(np.argwhere(tpr >= 0.99)[0])
                fpr95 = fpr[th_95]
                fpr99 = fpr[th_99]
                print('[ {} ]: FPR95: {} at th: {}'.format(dataset_key, fpr95, ths[th_95]))
                print('[ {} ]: FPR99: {} at th: {}'.format(dataset_key, fpr99, ths[th_99]))
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

    def thresholding(self, global_model):
        
        logging.info("################ Threshold Search #################")
        self.model.load_state_dict(global_model, strict=False)
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
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(x)
                for i in range(len(x)):
                    anomaly_map_i = anomaly_map[i][0]
                    for th_ in ths:
                        fpr = (np.count_nonzero(anomaly_map_i > th_) * 100) / im_scale
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
        logging.info(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        return best_th

    def object_localization(self, global_model, th=0):
        
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
            logging.info(f"#################################### {dataset_key} ############################################")

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
                
                masks = data[1][:, 0, :, :].view(nr_batches, 1, width, height).to(self.device)\
                    if masks_bool else None
                neg_masks = data[1][:, 1, :, :].view(nr_batches, 1, width, height).to(self.device)
                neg_masks[neg_masks>0.5] = 1
                neg_masks[neg_masks<1] = 0

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))

                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    anomaly_map_i = anomaly_map[i][0]
                    anomaly_score_i = anomaly_score[i][0]
                    mask_ = masks[i][0].cpu().detach().numpy() if masks_bool else None
                    neg_mask_ = neg_masks[i][0].cpu().detach().numpy() if masks_bool else None
                    bboxes = cv2.cvtColor(neg_mask_*255, cv2.COLOR_GRAY2RGB)
                    
                    cnts_gt = cv2.findContours((mask_*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []
                    for c_gt in cnts_gt:
                        xpos, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([xpos, y, x+w, y+h])
                        cv2.rectangle(bboxes, (xpos, y), (xpos + w, y + h), (0, 255, 0), 1)
                    
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    x_combo = copy.deepcopy(anomaly_map_i)
                    x_combo[x_combo < th] = 0

                    x_pos = x_combo * mask_
                    x_neg = x_combo * neg_mask_
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)
                    

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

                    ious = [np.round(res_anomaly,2), np.round(res_healthy,2)]

                    if (idx % 5) == 0: 
                        to_visualize = [
                            {'title': 'x', 'tensor': x_},
                            {'title': 'x_rec', 'tensor': x_rec_},
                            {'title': f'Anomaly  map {anomaly_map_i.max():.3f}', 'tensor': anomaly_map_i,
                             'cmap': 'plasma', 'vmax': anomaly_map_i.max()},
                            {'title': f'Combo map {x_combo.max():.3f}', 'tensor': x_combo,
                             'cmap': 'plasma', 'vmax': x_combo.max()}
                        ]

                        if 'mask' in x_rec_dict.keys():
                            masked_input = x_rec_dict['mask'] + x
                            masked_input[masked_input > 1] = 1
                            to_visualize.append(
                                {'title': 'Rec Orig', 'tensor': x_rec_dict['x_rec_orig'], 'cmap': 'gray'})
                            to_visualize.append({'title': 'Res Orig', 'tensor': x_rec_dict['x_res'], 'cmap': 'plasma',
                                                 'vmax': x_rec_dict['x_res'].max()})
                            to_visualize.append({'title': 'Mask', 'tensor': masked_input, 'cmap': 'gray'})

                        if masks_bool:
                            to_visualize.append({'title': 'GT', 'tensor': bboxes.astype(np.int64), 'vmax': 1})
                            to_visualize.append({'title': f'{res_anomaly}, TP: {tp}', 'tensor': x_pos,
                                                 'vmax': anomaly_map_i.max(), 'cmap': 'plasma'})
                            to_visualize.append({'title': f'{res_healthy}, FP: {fp}', 'tensor': x_neg,
                                                 'vmax': anomaly_map_i.max(), 'cmap': 'plasma'})

                            self._log_visualization(to_visualize, dataset_key, count)


            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                if metric == 'TP':
                    logging.info(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
                if metric == 'FP':
                    logging.info(f'FP: {np.sum(test_metrics[metric])} missed')
                metrics[metric].append(test_metrics[metric])

            logging.info("################################################################################")

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
