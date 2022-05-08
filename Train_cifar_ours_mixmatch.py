from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import pdb
import wandb
from typing import List, Optional, Tuple, Union, cast
import torchvision

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--adv', action="store_true")
parser.add_argument('--bound', default=0.02, type=float, help='bound for adversarial')
parser.add_argument('--num_iterations', default=5, type=int, help='eps for adversarial')
parser.add_argument('--lam', default=1, type=float, help='bound for adversarial')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
mu_cifar100 = torch.tensor(cifar100_mean).view(3,1,1)
std_cifar100 = torch.tensor(cifar100_std).view(3,1,1)


def reconst_images(x_adv, strong_x):
    grid_X = torchvision.utils.make_grid(strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid_AdvX = torchvision.utils.make_grid(x_adv[:10].data, nrow=10, padding=2, normalize=True)
    grid_Delta = torchvision.utils.make_grid(x_adv[:10]-strong_x[:10].data, nrow=10, padding=2, normalize=True)
    grid = torch.cat((grid_X, grid_AdvX, grid_Delta), dim=1)
    wandb.log({"Batch.jpg": [
        wandb.Image(grid)]}, commit=False)
    wandb.log({'His/l2_norm': wandb.Histogram((x_adv - strong_x).reshape(strong_x.shape[0], -1).norm(dim=1).cpu().detach().numpy(), num_bins=512),
               }, commit=False)


def normalize_flatten_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
) -> torch.Tensor:

    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_attack(model, inputs, targets_u, y_ori, flat_feat_ori, args):
    upper_limit = ((1 - mu_cifar100) / std_cifar100).cuda()
    lower_limit = ((0 - mu_cifar100) / std_cifar100).cuda()

    perturbations = torch.zeros_like(inputs)
    perturbations.uniform_(-0.01, 0.01)
    perturbations.data = clamp(perturbations, lower_limit - inputs, upper_limit - inputs)
    perturbations.requires_grad = True
    for attack_iter in range(args.num_iterations):
        # Decay step size, but increase lambda over time.
        step_size = \
            args.bound * 0.1 ** (attack_iter / args.num_iterations)
        lam = \
            args.lam * 0.1 ** (1 - attack_iter / args.num_iterations)

        if perturbations.grad is not None:
            perturbations.grad.data.zero_()

        inputs_adv = inputs + perturbations

        logits_adv, feat_adv = model(inputs_adv, adv=True, return_features=True)
        prob_adv = torch.softmax(logits_adv, dim=-1)
        y_adv = torch.log(torch.gather(prob_adv, 1, targets_u.view(-1, 1)).squeeze(dim=1))

        pip = (normalize_flatten_features(feat_adv) - \
        normalize_flatten_features(flat_feat_ori)).norm(dim=1).mean()
        constraint = y_ori - y_adv
        loss = -pip + lam * F.relu(constraint - args.bound).mean()
        loss.backward()

        grad = perturbations.grad.data
        grad_normed = grad / \
                      (grad.reshape(grad.size()[0], -1).norm(dim=1)
                       [:, None, None, None] + 1e-8)
        with torch.no_grad():
            y_after = torch.log(torch.gather(torch.softmax(
                         model(inputs + perturbations - grad_normed * 0.1, adv=True), dim=1),
                         1, targets_u.view(-1, 1)).squeeze(dim=1))
            dist_grads = torch.abs( y_adv - y_after
                         ) / 0.1
            norm = step_size / (dist_grads + 1e-4)
        perturbation_updates = -grad_normed * norm[:, None, None, None]

        perturbations.data = clamp(perturbations + perturbation_updates,
                                   lower_limit - inputs, upper_limit - inputs).detach()

    inputs_adv = (inputs + perturbations).detach()
    return inputs_adv


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            confidence, _ = torch.max(pu, dim=-1)
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
            # adv
            inputs_ori = torch.cat([inputs_u, inputs_u2], dim=0)
            targets_ori = torch.cat([targets_u, targets_u], dim=0)

            logits_ori, feat_ori = net(inputs_ori, adv=True, return_features=True)
            _, targets_uadv = torch.max(logits_ori, 1)
            prob = torch.softmax(logits_ori, dim=-1)
            y_w = torch.log(torch.gather(prob, 1, targets_uadv.view(-1, 1)).squeeze(dim=1))

        inputs_adv = get_attack(net, inputs_ori, targets_uadv, y_w, feat_ori, args)
        optimizer.zero_grad()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        # adv mixmatch
        all_inputsadv = torch.cat([inputs_x, inputs_x2, inputs_adv], dim=0)
        all_targetsadv = torch.cat([targets_x, targets_x, targets_ori], dim=0)

        idx = torch.randperm(all_inputsadv.size(0))

        input_aadv, input_badv = all_inputsadv, all_inputsadv[idx]
        target_aadv, target_badv = all_targetsadv, all_targetsadv[idx]

        mixed_inputadv = l * input_aadv + (1 - l) * input_badv
        mixed_targetadv = l * target_aadv + (1 - l) * target_badv

        logits_adv, feat_adv = net(mixed_inputadv[batch_size*2:], adv=True, return_features=True)
        prob_adv = torch.softmax(logits_adv, dim=-1)
        y_adv = torch.log(torch.gather(prob_adv, 1, targets_uadv.view(-1, 1)).squeeze(dim=1))
        Ladv = torch.mean((prob_adv - mixed_targetadv[batch_size*2:])**2)
        pip = (normalize_flatten_features(feat_adv) - \
               normalize_flatten_features(feat_ori).detach()).norm(dim=1)

        # mixmatch
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * (Lu+Ladv)  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()
        wandb.log({'loss/l_adv': Ladv.data.item(),
                   'loss/l_u': Lu.data.item(),
                   'loss /l_x': Lx.data.item(),
                   'confidence': confidence.mean().cpu().detach().numpy(),
                   'confidence_his': wandb.Histogram(confidence.cpu().detach().numpy(), num_bins=512),
                     'y_adv': y_adv.mean().cpu().detach().numpy(),
                     'y_w': y_w.mean().cpu().detach().numpy(),
                     'pip': pip.mean().cpu().detach().numpy(),
                     'l2_norm': torch.mean((inputs_adv - inputs_ori).reshape(inputs_ori.shape[0], -1).norm(dim=1)),
                     'l2_norm_his': wandb.Histogram((inputs_adv - inputs_ori).reshape(inputs_ori.shape[0], -1).norm(dim=1).cpu().detach().numpy(), num_bins=512)})
    reconst_images(inputs_adv, inputs_ori)

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()
    wandb.log({'test/acc': acc},commit=False)

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model(args):
    if args.adv:
        model = ResNet18(num_classes=args.num_class, bn_adv_flag=True, bn_adv_momentum=0.01)
    else:
        model = ResNet18(num_classes=args.num_class)

    model = model.cuda()
    return model

stats_log=open('./checkpoint/advmix%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w')
test_log=open('./checkpoint/advmix%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30
wandb.init(config=args)
loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model(args)
net2 = create_model(args)
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         

    test(epoch,net1,net2)  


