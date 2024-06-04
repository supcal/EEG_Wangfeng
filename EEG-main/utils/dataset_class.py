import logging
from torch.utils.data import Dataset


class SEEDDataset(Dataset):
    def __init__(self, X, y):
        '''
        SEEDDataset
        '''
        self.X = X.float()
        self.y = y+1
        print((self.y == 0).sum(), (self.y == 1).sum(),
              (self.y == 2).sum(), len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = int(self.y[index])
        return X, y


class SEEDAdjDataset(Dataset):
    def __init__(self, X, y):
        '''
        SEEDDataset
        '''
        self.X = X
        self.y = y
        # print((self.y == 0).sum(), (self.y == 1).sum(),
        #       (self.y == 2).sum(), len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        data = X['data'].float()
        coh_adj = X['coh_adj'].float()
        pcc_adj = X['pcc_adj'].float()
        plv_adj = X['plv_adj'].float()
        nmi_adj = X['nmi_adj'].float()

        class_y = int(self.y[index, 0]+1)
        domain_y = int(self.y[index, 1])
        return (data, coh_adj, pcc_adj, plv_adj, nmi_adj), (class_y, domain_y)


class DEAPDataset(Dataset):
    def __init__(self, feature, label):
        '''

        '''

        self.feature = feature
        self.label = (label > 5)
        print((self.label == 0).sum(), (self.label == 1).sum(), len(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        features = self.feature[index].float()
        label = self.label[index]
        label = int(label)
        return features, label


class AMIGOSDataset(Dataset):
    def __init__(self, feature, label):
        '''

        '''

        self.feature = feature
        self.label = label
        print((self.label == 0).sum(), (self.label == 1).sum(), len(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        features = self.feature[index].float()
        label = self.label[index]
        label = int(label)
        return features, label


class SEEDIVDataset_spp(Dataset):
    def __init__(self, X, y):
        '''
        SEEDIVDataset_spp
        '''
        self.X = X
        self.y = y
        print((self.y == 0).sum(), (self.y == 1).sum(),
              (self.y == 2).sum(), (self.y == 3).sum(), len(self.y))
        logging.info(
            f'label0: {(self.y == 0).sum()}, label1: {(self.y == 1).sum()}, label2: {(self.y == 2).sum()}, label3: { (self.y == 3).sum()}, length: {len(self.y)}')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index].float()
        y = int(self.y[index])
        return X, y


class SEEDIVAdjDataset(Dataset):
    def __init__(self, X, y):
        '''
        SEEDIVAdjDataset
        '''
        self.X = X
        self.y = y
        # print((self.y == 0).sum(), (self.y == 1).sum(),
        #       (self.y == 2).sum(), len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        data = X['data'].float()
        coh_adj = X['coh_adj'].float()
        pcc_adj = X['pcc_adj'].float()
        plv_adj = X['plv_adj'].float()
        nmi_adj = X['nmi_adj'].float()

        class_y = int(self.y[index, 0])
        domain_y = int(self.y[index, 1])
        return (data, coh_adj, pcc_adj, plv_adj, nmi_adj), (class_y, domain_y)


class FACEDataset(Dataset):
    def __init__(self, X, y):
        '''
        FACEDataset
        '''
        self.X = X
        self.y = y
        print((self.y == 0).sum(),
              (self.y == 1).sum(),
              (self.y == 2).sum(),
              len(self.y))
        logging.info(
            f'label0: {(self.y == 0).sum()},label1: {(self.y == 1).sum()},label2: {(self.y == 2).sum()},length: {len(self.y)}')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index].float()
        y = int(self.y[index])
        return X, y
