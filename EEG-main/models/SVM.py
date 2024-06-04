import torch.nn as nn
import copy

class SVM(nn.Module):
    def __init__(self,  args, device='cpu'):
        super(SVM, self).__init__()
        num_classes = args.nclass
        input_size = args.channels_num* args.feature_len*5
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        tsne1 = copy.deepcopy(x.detach())
        x=x.reshape(x.shape[0],-1)
        tsne2 = copy.deepcopy(x.detach())
        return self.linear(x),tsne1,tsne2
    
    def loss(self, model, pred, label):
        focal = nn.CrossEntropyLoss()(pred[0], label)
        return focal