
import os


class DownstreamEvaluator(object):
    

    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        
        self.name = name
        self.model = model.to(device)
        self.device = device
        self.test_data_dict = test_data_dict
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.image_path = checkpoint_path + '/images/'
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        super(DownstreamEvaluator, self).__init__()

    def start_task(self, global_model):
        
        raise NotImplementedError("[DownstreamEvaluator::start_task]: Please Implement start_task() method")
