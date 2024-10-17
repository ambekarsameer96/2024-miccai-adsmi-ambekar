import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch
import torch.nn as nn























        


        


        


        




























class mahalanobis_loss(nn.Module):
    def __init__(self):
        super(mahalanobis_loss, self).__init__()
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, input_data, reconstruction):
        
        input_data = input_data.to(self.device)
        reconstruction = reconstruction.to(self.device)

        
        input_data_flat = input_data.view(input_data.size(0), -1)
        reconstruction_flat = reconstruction.view(reconstruction.size(0), -1)

        
        residual = input_data_flat - reconstruction_flat

        
        covariance_matrix = torch.matmul(residual.t(), residual) / input_data_flat.size(0)

        
        inv_covariance_matrix = torch.pinverse(covariance_matrix + 1e-8 * torch.eye(residual.size(1)).to(self.device))

        
        mahalanobis_loss = torch.sum(torch.matmul(torch.matmul(residual, inv_covariance_matrix), residual.t())) / input_data_flat.size(0)

        return mahalanobis_loss