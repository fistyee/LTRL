import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
eps = 1e-7 


def soft_cross_entropy_with_logits(logits, soft_targets):
    # 先计算softmax以获得概率
    probs = F.softmax(logits, dim=1)
    
    # 计算softlabel交叉熵
    loss = -(soft_targets * torch.log(probs)).sum(dim=1).mean()
    
    return loss

def convert_to_soft_labels(labels, similarity_matrix, num_classes, epoch):
    """
    Convert hard labels to soft labels using a similarity matrix or default to one-hot encoding.

    Args:
        labels (Tensor): Tensor containing the hard labels (shape: [batch_size]).
        similarity_matrix (Tensor, optional): Tensor containing the similarity matrix (shape: [num_classes, num_classes]).
        num_classes (int): Number of classes.
        epoch (int): Current epoch number, used to compute alpha.
        
    Returns:
        Tensor: Tensor containing the soft labels (shape: [batch_size, num_classes]).
    """
    batch_size = labels.size(0)
    smoothing = 0.1
    alpha = 0.01
    # Create an empty tensor for the soft labels
    soft_labels = torch.zeros(batch_size, num_classes, device=labels.device)
    
    if similarity_matrix is not None:
        for i in range(batch_size):
            label = labels[i].item()  # Get the hard label
            # Get similarity scores for this label with all other labels
            similarity_scores = similarity_matrix[label]
            soft_label_probs = F.softmax(similarity_scores, dim=0)

            # Apply softmax to convert similarity scores to probabilities
            soft_labels[i] = (1 - alpha) * F.one_hot(labels[i], num_classes=num_classes).float() + alpha * soft_label_probs
    else:
        # Create one-hot encoded labels
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        
        # Apply label smoothing
        soft_labels = one_hot * (1 - smoothing) + (smoothing / num_classes)
    
    return soft_labels

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target, extra_info= None): # output is logits
        #pdb.set_trace()
        if extra_info is None:
            return F.cross_entropy(output_logits, target) 
        return F.cross_entropy(output_logits, target)#, weight=self.per_cls_weights),

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index
         
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss
  
 
class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy 
        #self.base_loss = soft_CE
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 
        self.many = []
        self.few = []


    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0
        
        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  
        
        pdb.set_trace()
        # Softmax loss for expert 1 
        loss += self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        # Inverse Softmax loss for expert 3 
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss += self.base_loss(expert3_logits, target)
    
        return loss
    
class DiverseExpertLoss_RL(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2):
        super().__init__()
        #self.base_loss = F.cross_entropy
        self.base_loss = soft_cross_entropy_with_logits
        
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.p = torch.tensor(np.array(cls_num_list)).float().cuda()
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 
        self.sum = torch.zeros(self.C_number)
        self.sum = self.sum.cuda()
        self.many = []
        self.few = []

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0
        #if extra_info['epoch'] > 1:
        #    pdb.set_trace()
        label = convert_to_soft_labels(target, extra_info['cos_feature'], len(self.p), extra_info['epoch'])

        # parameters
        temperature = 2
        old_pred = extra_info['old_pred']
        old_pred = old_pred.to(target.device)

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  


        # Softmax loss for expert 1 
        loss += self.base_loss(expert1_logits, label)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, label)
        
        # Inverse Softmax loss for expert 3 
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss += self.base_loss(expert3_logits, label)


        # RL
        # 统计 target 中每个元素的计数 按照 target 中的索引取出对应的值
        num = torch.bincount(target, minlength=len(self.p)).float()[target]
        # 扩展维度并复制
        num = num.unsqueeze(1).repeat(1, len(self.p)).to(target.device)                                                                           
        expert_sum_logits = extra_info['output'] / num

        #forward
        kl_loss = 0
        teacher1_max, teacher1_index = torch.max(F.softmax((old_pred), dim=1).detach(), dim=1)  
        teacher_softmax = F.softmax((old_pred)/temperature, dim=1) 
        student_softmax = F.log_softmax(expert_sum_logits/temperature, dim=1)
        if torch.sum((teacher1_index==target))>0:
            kl_loss = kl_loss + F.kl_div(student_softmax[(teacher1_index== target)], teacher_softmax[(teacher1_index== target)], reduction='batchmean')  * (temperature**2) 
            

        kl_loss = kl_loss*3
        # return loss
        return loss, kl_loss, expert_sum_logits.cpu().detach()
    

class RIDELoss_RL(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        #self.base_loss = F.cross_entropy
        self.base_loss = soft_cross_entropy_with_logits
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight
        self.p = torch.tensor(np.array(cls_num_list)).float().cuda()
        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0
        label = convert_to_soft_labels(target, extra_info['cos_feature'], len(self.p), extra_info['epoch'])

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, label)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                self.base_loss = F.cross_entropy
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')


        # parameters

        temperature = 2

        old_pred = extra_info['old_pred']
        old_pred = old_pred.to(target.device)

        # RL
        expert_sum_logits = extra_info['output']
        num = torch.bincount(target, minlength=len(self.p)).float()[target]
        # 扩展维度并复制
        num = num.unsqueeze(1).repeat(1, len(self.p)).to(target.device)                                                                           
        expert_sum_logits = extra_info['output'] / num

        kl_loss = 0

        teacher1_max, teacher1_index = torch.max(F.softmax((old_pred), dim=1).detach(), dim=1)  
        teacher_softmax = F.softmax((old_pred)/temperature, dim=1) 
        student_softmax = F.log_softmax(expert_sum_logits/temperature, dim=1)
        if torch.sum((teacher1_index==target))>0:
            kl_loss = kl_loss + F.kl_div(student_softmax[(teacher1_index== target)], teacher_softmax[(teacher1_index== target)], reduction='batchmean')  * (temperature**2) 
            
        kl_loss = kl_loss*3
        return loss, kl_loss, expert_sum_logits.cpu().detach()



class CrossEntropyLoss_RL(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        reweight_CE = True
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None
        self.many = []
        self.few = []
        self.sum = torch.zeros(10)
        self.sum = self.sum.cuda()
        self.p = torch.tensor(np.array(cls_num_list)).float().cuda()
        self.base_loss = soft_cross_entropy_with_logits

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target, extra_info=None): # output is logits
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction
        loss = 0
        
        label = convert_to_soft_labels(target, extra_info['cos_feature'], len(self.p), extra_info['epoch'])
        loss = self.base_loss(output_logits + torch.log(self.p+1e-9), label)
        
        old_pred = extra_info['old_pred']
        old_pred = old_pred.to(target.device)
        
        #pdb.set_trace()
       
        expert_sum_logits = output_logits.clone() #/num #+ torch.log(self.p) #* self.per_cls_weights

        expert_softmax = F.softmax((expert_sum_logits), dim=1)
        old_pred_softmax = F.softmax(old_pred, dim = 1)
    
        batch_idx = extra_info['index']

        # RL
        temperature = 2

        num = torch.bincount(target, minlength=len(self.p)).float()[target]
        # 扩展维度并复制
        num = num.unsqueeze(1).repeat(1, len(self.p)).to(target.device)                                                                           
        expert_sum_logits = extra_info['output'] / num

        #forward
        kl_loss = 0
        teacher1_max, teacher1_index = torch.max(F.softmax((old_pred), dim=1).detach(), dim=1)  
        teacher_softmax = F.softmax((old_pred)/temperature, dim=1) 
        student_softmax = F.log_softmax(expert_sum_logits/temperature, dim=1)
        if torch.sum((teacher1_index==target))>0:
            kl_loss = kl_loss + F.kl_div(student_softmax[(teacher1_index== target)], teacher_softmax[(teacher1_index== target)], reduction='batchmean')  * (temperature**2) 
            
        return loss, kl_loss, expert_sum_logits.cpu().detach()
