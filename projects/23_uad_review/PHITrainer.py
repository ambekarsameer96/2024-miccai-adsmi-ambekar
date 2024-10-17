from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
from optim.custom_losses import mahalanobis_loss
import matplotlib.pyplot as plt
import copy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from torch.optim import Adam
from torch.autograd import Variable


import rich



from core.custom_data import s2_CustomDataset
from torch.utils.data import DataLoader as torch_dataloader

from phi_model import phi_utils
class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.optimizer_phi = Adam(model.PHI.parameters(), lr=training_params['optimizer_params']['lr'])
        
        path_to_s2 =training_params['path_to_s2']
        self.ra_weights = training_params['path_to_ra_weights']
        self.s2_dataset = s2_CustomDataset(csv_file=path_to_s2)
        self.s2_dataloader = torch_dataloader( self.s2_dataset,shuffle=True, drop_last=True,pin_memory=True, batch_size=8)

        self.s2_data_iterator = iter(self.s2_dataloader)
        self.MSE = torch.nn.MSELoss()
        
        
        
        
        
        
        
        self.content_weight = float(training_params['content_weight'])
        self.style_weight = float(training_params['style_weight']) 
        self.tv_weight = float(training_params['tv_weight']) 
        
        self.mahalanobis_weight = 1
        self.mahalanobis_loss_obj = mahalanobis_loss()

        
        
        
        
        self.optimizing_img = None
        

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        
        if model_state is not None:
            self.model.load_state_dict(model_state)  
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  
        
        epoch_losses = []
        epoch_losses_pl = []
        
        self.early_stop = False

        ## LOAD DOMAIN 2 FROM DISK
        
        ckpt = torch.load(self.ra_weights)
        ckpt_model = ckpt['model_weights']
        

        self.model.RA.load_state_dict(ckpt_model, strict=False)
        print('=========Loaded weights============')
        path_to_domain = ''
        domain_2_images = []
        for path in path_to_domain:
            Im.read(path)
            domain_2_images.append()

        
        
        
        

        

        
        
        

        for epoch in range(self.training_params['nr_epochs']):
        
        
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_pl, count_images = 1.0, 1.0, 0
            content_loss_tot = 0.0
            style_loss_tot = 0.0
            ssim_loss_tot = 0.0
            
            self.epoch = epoch

            for data in self.train_ds:
                
                images = data[0].to(self.device)
                
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape
                count_images += b

                for param in self.model.RA.parameters():
                    param.requires_grad = False

                


                    
                
                self.optimizer_phi.zero_grad()
                
                try: 
                    style_images = next(self.s2_data_iterator)
                except:
                    self.s2_data_iterator = iter(self.s2_dataloader)
                    style_images = next(self.s2_data_iterator)

                style_images = style_images.to(self.device)
                
                if epoch==0:
                    init_img = style_images[0]
                    
                    init_img = init_img.to(self.device)
                    
                    
                    
                    
                    init_img = init_img.repeat(1, 3, 1, 1)
                    

                    
                    
                    



                    self.optimizing_img = Variable(init_img, requires_grad=True)
                    
                    self.VGG_optimizer = Adam([self.optimizing_img], lr=1e1)
                
                
                x_s1_, x_s1_phi, x_s2_re, f_result = self.model(transformed_images, style_images)


                
                
                
                
                
                
                
                
                
                x_s1_rep = x_s1_.repeat(1, 3, 1, 1)
                
                style_images_rep = style_images.repeat(1, 3, 1, 1)

                content_x_s1_ =  self.model.VGG(transformed_images.repeat(1,3,1,1)) 
                style_x_s2 =  self.model.VGG(style_images_rep) 
                
                x_s1_phi_rep = x_s1_phi.repeat(1, 3, 1, 1)
                
                
                actual_content = self.model.VGG(x_s1_phi_rep)
                
                

                
                
                

                
                

                
                

                
                
                
                
                
                

                
                

                
                
                
                
                
                
                
                self.optimizer_phi.zero_grad()
                self.actual_content_representation = actual_content[self.model.content_feature_maps_index_name[0]].squeeze(axis=0)
                
                
                self.actual_style_representation = [phi_utils.gram_matrix(x) for cnt, x in enumerate(actual_content) if cnt in self.model.style_feature_maps_indices_names[0]]
                
                
                self.target_content_representation = content_x_s1_[self.model.content_feature_maps_index_name[0]].squeeze(axis=0)
                
                
                self.target_style_representation = [phi_utils.gram_matrix(x) for cnt, x in enumerate(style_x_s2) if cnt in self.model.style_feature_maps_indices_names[0]]
                
                
                loss_content = torch.nn.MSELoss(reduction='mean')(self.actual_content_representation, self.target_content_representation)
                
                
                loss_style = 0.0
                
                loss_ssim = 1 - ssim(x_s1_phi, style_images, data_range=1.0, size_average=True)

                
                
                for gram_gt, gram_hat in zip(self.actual_style_representation, self.target_style_representation):
                    loss_style += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
                loss_style /= len(self.target_style_representation)
                
                
                
                total_loss = (self.content_weight * loss_content) + (self.style_weight * loss_style) 
                total_loss.backward()
                self.optimizer_phi.step()
                
                
                
                
                
                
                
                
                
                

                
                
                
                
                
                

                
                loss_rec = self.criterion_rec(x_s1_phi, images, f_result)
                
                

                loss_pl = self.criterion_PL(x_s1_phi, style_images)
                
                
                

                
                

                

                
                
                
                
                
                
                
                

                
                
                
                batch_loss += loss_rec.item() * images.size(0)
                batch_loss_pl += loss_pl.item() * images.size(0)
                content_loss_tot += loss_content.item() * images.size(0)
                style_loss_tot += loss_style.item() * images.size(0)
                ssim_loss_tot += loss_ssim.item() * images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_pl
            epoch_losses.append(epoch_loss)
            epoch_losses_pl.append(epoch_loss_pl)
            
            
            content_loss_count = content_loss_tot / count_images if count_images > 0 else content_loss_tot
            style_loss_count = style_loss_tot / count_images if count_images > 0 else style_loss_tot
            ssim_loss_count = ssim_loss_tot / count_images if count_images > 0 else ssim_loss_tot

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_PL_": epoch_loss_pl, '_step_': epoch})
            
            
            
            wandb.log({"Train/Content_loss": content_loss_count, '_step_': epoch})
            wandb.log({"Train/Style_loss": style_loss_count, '_step_': epoch})
            wandb.log({"Train/SSIM_loss": ssim_loss_count, '_step_': epoch})

            
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            img = transformed_images[0].cpu().detach().numpy()
            img_s2 = style_images[0].cpu().detach().numpy()

            
            rec = x_s1_[0].cpu().detach().numpy()
            rec2 = x_s1_phi[0].cpu().detach().numpy()
            x_s2_re_rec3 = x_s2_re[0].cpu().detach().numpy()
            
            
            elements = [img, img_s2, rec, x_s2_re_rec3, rec2, np.abs(rec - img), np.abs(rec2-img)]
            names  = ['meta_source_img', 'meta_target_image', 'reconstruct_meta_source', 'reconsutruct_meta_target', 'PHI_meta_target', 'anomaly_1__'+ str(np.round(np.mean(np.abs(rec - img)),3)), 'anomaly_2__' + str(np.round(np.mean(np.abs(rec2 - img)),3))]
            v_maxs = [1, 1,1,1,1, 0.5, 0.5]
            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 4)
            for i in range(len(axarr)):
                axarr[i].axis('off')
                v_max = v_maxs[i]
                c_map = 'gray' if v_max == 1 else 'inferno'
                axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)
                
                
                axarr[i].set_title(names[i])





            wandb.log({'Train/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

            
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                
                x_, x_s2_,_, x_rec = self.test_model(x, x)
                loss_rec = self.criterion_rec(x_, x, x_rec)
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

        img = x.detach().cpu()[0].numpy()
        rec = x_.detach().cpu()[0].numpy()

        elements = [img, rec, np.abs(rec - img)]
        v_maxs = [1, 1, 0.5]
        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)
        for i in range(len(axarr)):
            axarr[i].axis('off')
            v_max = v_maxs[i]
            c_map = 'gray' if v_max == 1 else 'inferno'
            axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)

        wandb.log({task + '/Example_': [
            wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_rec'] / test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                           self.client_path + '/best_model.pt')
                self.best_weights = copy.deepcopy(model_weights)
                self.best_opt_weights = copy.deepcopy(opt_weights)
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)



    def build_loss(self):
        target_content_representation = self.target_representations[0]
        target_style_representation = self.target_representations[1]

        current_set_of_feature_maps = self.model.VGG(self.optimizing_img)

        self.content_feature_maps_index = self.model.content_feature_maps_index_name[0]
        self.style_feature_maps_indices = self.model.style_feature_maps_indices_names[0]
        
        

        current_content_representation = current_set_of_feature_maps[self.content_feature_maps_index].squeeze(axis=0)
        
        
        
        
        content_loss = torch.nn.MSELoss(reduction='mean')(self.target_representations[0] , self.target_representations[1])
        

        style_loss = 0.0
        current_style_representation = [phi_utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in self.style_feature_maps_indices]
        for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
            style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        style_loss /= len(target_style_representation)

        tv_loss = phi_utils.total_variation(self.optimizing_img)

        
        
        

        
        
        total_loss = (self.content_weight * content_loss) + (self.style_weight * style_loss)
        
        
        
        wandb.log ({'phi_content_loss': content_loss, 'phi_style_loss': style_loss, 'phi_tv_loss': tv_loss, 'phi_total_loss': total_loss, '_step_': self.epoch})

        return total_loss, content_loss, style_loss, tv_loss

    
    def tuning_step(self):
        self.content_feature_maps_index = self.model.content_feature_maps_index_name[0]
        self.style_feature_maps_indices = self.model.style_feature_maps_indices_names[0]

        
        total_loss, content_loss, style_loss, tv_loss = self.build_loss()
        
        self.optimizer_phi.zero_grad()
        
        total_loss.backward(retain_graph=True)
        
        
        
        
        self.optimizer_phi.step()
        

        
        


        return total_loss, content_loss, style_loss, tv_loss




