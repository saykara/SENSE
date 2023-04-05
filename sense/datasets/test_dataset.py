import torch.utils.data as data



class Stage1TestDataset(data.Dataset):
    def __init__(self, root, path_list, flow_loader, disp_loader, crop_transform, transform
        ):
        super(Stage1TestDataset, self).__init__()
        self.root = root
        self.path_list = path_list
        self.flow_loader = flow_loader
        self.disp_loader = disp_loader
        self.crop_transform = crop_transform
        self.transform = transform

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        # load image
        # ims = [cv2.imread(n).astype(np.float32) / 255.0 for n in inputs]
        flow_ims, flow_gt = self.flow_loader(self.root, inputs[:2], targets[:2])
        disp_ims, disp_gt = self.disp_loader(self.root, inputs[2:], targets[2:])

        if self.crop_transform is not None:
            flow_ims, flow_gt = self.crop_transform(flow_ims, flow_gt)
        if self.crop_transform is not None:
            disp_ims, disp_gt = self.crop_transform(disp_ims, disp_gt)
            
        if self.transform is not None :
            flow_gt = self.transform(flow_gt)
        if self.transform is not None :
            disp_gt = self.transform(disp_gt)
            
        flow_ims_new = []
        disp_ims_new = []
        if self.transform is not None:
            for i in range(len(flow_ims)):
                # flow_ims[i] = self.transform_additional(flow_ims[i])  
                flow_ims_new.append(self.transform(flow_ims[i]))
            for i in range(len(disp_ims)):
                # disp_ims[i] = self.transform_additional(disp_ims[i])  
                disp_ims_new.append(self.transform(disp_ims[i]))
        return flow_ims_new, flow_gt, disp_ims_new, disp_gt

    def __len__(self):
        return len(self.path_list)


class Stage2TestDataset(data.Dataset):
    def __init__(self, root, path_list, crop_transform, transform, imread
        ):
        super(Stage1TestDataset, self).__init__()
        self.root = root
        self.path_list = path_list
        self.imread = imread
        self.crop_transform = crop_transform
        self.transform = transform

    def __getitem__(self, index):
        cur_l, nxt_l = self.path_list[index]
        
        if self.crop_transform is not None:
            flow_ims, flow_gt = self.crop_transform(flow_ims, flow_gt)
        if self.crop_transform is not None:
            disp_ims, disp_gt = self.crop_transform(disp_ims, disp_gt)
        
        if self.transform is not None :
            flow_gt = self.transform(flow_gt)
        if self.transform is not None :
            disp_gt = self.transform(disp_gt)
            
        flow_ims_new = []
        disp_ims_new = []
        if self.transform is not None:
            for i in range(len(flow_ims)):
                # flow_ims[i] = self.transform_additional(flow_ims[i])  
                flow_ims_new.append(self.transform(flow_ims[i]))
            for i in range(len(disp_ims)):
                # disp_ims[i] = self.transform_additional(disp_ims[i])  
                disp_ims_new.append(self.transform(disp_ims[i]))
        return flow_ims_new, flow_gt, disp_ims_new, disp_gt

    def __len__(self):
        return len(self.path_list)