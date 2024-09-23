import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16
import model.model as module_arch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import pdb

def normalize_vector(tensor):
    norm = torch.norm(tensor)
    return tensor / norm if norm > 0 else tensor

def cosine_similarity(tensor1, tensor2):
    tensor1 = normalize_vector(tensor1)
    tensor2 = normalize_vector(tensor2)
    dot_product = torch.dot(tensor1, tensor2)
    # 确保结果在 [-1, 1] 范围内
    return torch.clamp(dot_product, min=-1.0, max=1.0)

def compute_cosine_similarities(median_vectors):
    num_vectors = len(median_vectors)
    similarities = torch.zeros((num_vectors, num_vectors), device='cuda')
    
    for i in range(num_vectors):
        for j in range(i, num_vectors):
            # 计算第 i 个和第 j 个向量之间的余弦相似度
            similarity = cosine_similarity(median_vectors[i], median_vectors[j])
            similarities[i, j] = similarity
            similarities[j, i] = similarity  # 余弦相似度矩阵是对称的
    
    return similarities

def compute_median_vector(all_feature):
    median_vectors = []
    
    for feature_list in all_feature:
        # 将每个列表中的 48 维向量堆叠成一个 tensor
        vectors = torch.stack(feature_list)
        
        # 计算每个维度的中值
        median_vector = torch.median(vectors, dim=0).values
        
        # 将中值向量添加到结果列表
        median_vectors.append(median_vector)
    
    return median_vectors

def get_gradient(model:nn.Module):
    gradient = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            grad = param.grad.clone().detach()
            gradient.append(grad.view(-1))

    return gradient    

def orthogonal_graident(grads_x, grads_y, epoch):
    orth_grads_y = []
    orth_grads_z = []
    cnt = 0
    grads = []
    sum_y = 0
    sum_oy = 0
    for x, y in zip(grads_x, grads_y):
        InP_xy = torch.matmul(y, x) 
        Inp_xx = torch.norm(x, p=2) ** 2
        oy = y - InP_xy/Inp_xx * x
        '''
        InP_yx = torch.matmul(x, y) 
        Inp_yy = torch.norm(y, p=2) ** 2
        ox = x - InP_yx/Inp_yy * y 
        '''
        length_y = torch.norm(y, p=2)
        length_oy = torch.norm(oy, p=2)
        sum_y = sum_y+length_y
        sum_oy = sum_oy+length_oy
        # 计算保留下来的分量占原始y的比例
        ratio_retained = float(length_oy / length_y)

        # 计算去除的分量占原始y的比例
        ratio_removed = 1 - ratio_retained
        grads.append(ratio_retained)
        
        if(InP_xy < 0):
            orth_grads_y.append(oy+x)
            cnt = cnt+1
        else:
            orth_grads_y.append(x+y)
    
    #if(epoch%50 == 0):
    #    print(grads)
    sumy = float(sum_oy / sum_y)
    cnt = cnt * 1.0/ len(grads_x)
    
    '''
    for x, oy, z in zip(grads_x, orth_grads_y, grads_z):
        
        InP_zx = torch.matmul(z, x)
        InP_xx = torch.norm(x, p=2) ** 2

        InP_zoy = torch.matmul(z, oy)
        InP_oyoy = torch.norm(oy, p=2) ** 2

        oz = z - InP_zx/InP_xx * x - InP_zoy/InP_oyoy * oy
        orth_grads_z.append(oz)
    '''
    return orth_grads_y,cnt,sumy


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)
        print("self.add_extra_info",self.add_extra_info)
       
        self.acc = []
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.predictions = torch.zeros(len(self.data_loader.dataset), len(self.data_loader.dataset.classes))
        # self.predictions = torch.zeros(len(self.data_loader.dataset), self.data_loader.num_classes)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # ------------------------------------------ #
        # KLD 初始化一个列表来存储每个epoch的logits和标签
        self.logits_history = []
        self.labels_history = []
        # ------------------------------------------ #


    def _train_epoch(self, epoch, cos_feature):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.real_model._hook_before_iter()
        self.train_metrics.reset()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)


        cnt = 0
        c = 0
        n = len(self.data_loader.dataset.classes)
        #pdb.set_trace()
        all_feature = [[] for _ in range(n)]  # 初始化包含 n+1 个空列表的列表
        for batch_idx, data in enumerate(self.data_loader):

            # data, target = data
            # ------------------------- #
            data, target, index = data
            # ------------------------- #
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                if self.real_model.requires_target:
                    output = self.model(data, target=target) 
                    output, loss = output   
                else:
                    extra_info = {}
                    output = self.model(data)
                    #pdb.set_trace()
                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                            feat = output["feat"]
                            extra_info.update({
                                "output": output["output"],
                                "logits": logits.transpose(0, 1),
                                "old_pred":self.predictions[index],
                                "index": batch_idx,
                                "epoch": epoch,
                                "cos_feature": cos_feature,
                            })
                        else:
                            feat = output["feat"]
                            extra_info.update({
                                "output": output["output"],
                                "index": batch_idx, 
                                "epoch": epoch,
                                "old_pred":self.predictions[index],
                                "cos_feature": cos_feature,
                                })
   
                    

                    if isinstance(output, dict):
                        output = output["output"]

                    if self.add_extra_info:
                        # KS
                        for i in range(len(target)):
                            all_feature[target[i]].append(feat[i])

                        loss, kl_loss, self.predictions[index] = self.criterion(output_logits=output, target=target, extra_info=extra_info)
                    else:
                        extra_info.update({
                                "old_pred":self.predictions[index],
                            })   
                        loss = self.criterion(output_logits=output, target=target)#, extra_info=extra_info) 
            if not use_fp16:
                ###

                if kl_loss:
                    loss.backward(retain_graph=True)
                    t_grads = get_gradient(self.model)
                     
                    self.optimizer.zero_grad()
                    kl_loss.backward()
                    n_grads = get_gradient(self.model)
                     
                    orth_grads, cnt, sumy = orthogonal_graident(t_grads, n_grads, epoch)
                    c = c + cnt
                    for idx, (_, param) in enumerate(self.model.named_parameters()):
                        if param.requires_grad:
                            param.grad = orth_grads[idx].view(param.size())
                    #loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()


                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, return_length=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.4f} min group LR: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        
        #KR
        median_feature = compute_median_vector(all_feature)
        cos_feature = compute_cosine_similarities(median_feature)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            self.acc.append(val_log['accuracy'])
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log, cnt, sumy, cos_feature.detach()

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()


        # KLD
        all_logits = []
        all_labels = []
        ######### 
        
        with torch.no_grad():
           
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if isinstance(output, dict):
                    output = output["output"]
                num_classes = len(self.data_loader.dataset.classes)
      
                one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
                one_hot.scatter_(1, target.unsqueeze(1), 1)
                loss = self.criterion(output, one_hot)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))



                # KLD
                all_logits.append(output)
                all_labels.append(target)
                #########

        # 将本epoch的logits和标签添加到历史记录中
        self.logits_history.append(torch.cat(all_logits))
        self.labels_history.append(torch.cat(all_labels))
        if len(self.logits_history) > 2:
            self.logits_history.pop(0)
            self.labels_history.pop(0)

        # if epoch > 190:
            # print(logits_average)

        # 计算KL散度
        # if epoch > 190:  
        #     kl_per_class = self.calculate_kl_div_per_class(self.logits_history[-2], self.logits_history[-1])
        #     print(f"KL divergence per class for epoch {epoch}: {kl_per_class}")
        #     self.plot_and_save_kl(kl_per_class, epoch)
        #     print(kl_per_class)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def calculate_kl_div_per_class(self, logits_prev, logits_current):
        num_classes = logits_prev.size(1)
        kl_per_class = torch.zeros(num_classes)

        for c in range(num_classes):
            idxs = (self.labels_history[0] == c)  # 假设labels的顺序在每个epoch中都是一样的，所以我们只使用一次labels_history
            if idxs.sum() > 0:  # 只对存在的类别计算
                kl_per_class[c] = F.kl_div(F.log_softmax(logits_prev[idxs], dim=1), F.softmax(logits_current[idxs], dim=1), reduction='sum').item()

        return kl_per_class

    def plot_and_save_kl(self, kl_per_class, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(kl_per_class, linestyle='-')
        plt.xlabel('Class')
        plt.ylabel('KL Divergence')
        plt.title(f'KL Divergence for Epoch {epoch}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        if not os.path.exists('fig_kl'):
            os.makedirs('fig_kl')

        plt.savefig(f'fig_kl/ce_cifar100_ir1/kl_epoch_{epoch}.png')
        plt.close()
