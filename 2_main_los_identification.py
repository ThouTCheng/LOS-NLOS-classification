#!/usr/bin/env python
# coding: utf-8
# -- coding: utf-8 -
#autor:zhouxiaoxing
'''
this script is the main function of LOS/NLOS identification
'''
# In[1]:
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import time
from warm_up_coslr import WarmupCosineLR
from model_zxx import Model_ericssion_one
import random
def set_random(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
def get_data_loader(path,batch_size):
    with open(path, 'rb') as f:
        X, y = pickle.load(f)
    print('data load done')
    pack_data = list(zip(X, y))
    random.shuffle(pack_data)
    print(type(pack_data), len(pack_data))
    print('1-5',y[1:5])
    train_loader = torch.utils.data.DataLoader(pack_data[0:90000], shuffle=False, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(pack_data[90000:95000], shuffle=False, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(pack_data[95000:99999],shuffle=False,batch_size=batch_size)
    # train_loader = torch.utils.data.DataLoader(pack_data[0:18000]+pack_data[20000:38000], shuffle=True, batch_size=batch_size)
    # valid_loader = torch.utils.data.DataLoader(pack_data[18000:19000]+pack_data[38000:39000], shuffle=True, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(pack_data[19000:20000]+pack_data[39000:40000],shuffle=True,batch_size=batch_size)
    return train_loader,valid_loader,test_loader
def model_train_process(train_loader,model,optimizer,scheduler,criterion,train_LOSS,lrs):
    running_loss,correct,total = 0,0,0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.float(), labels.float()
        # print('the input is ',inputs.shape,labels.shape)
        # print('len_train_loader',len(train_loader))
        # break
        output = model(inputs)
        optimizer.zero_grad()
        loss = criterion(output,labels.argmax(1))
        # loss = criterion(torch.squeeze(output),labels)
        # print('labels idx =1 station',)
        # print('output and labels is',output,labels)
        # print('loss is ',loss)
        # lab = labels.numpy()
        # print(lab)
        # los_idx = np.argwhere(lab==0)
        # pr_los = output[los_idx]
        # print('los idx is',los_idx,pr_los)
        # break
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (output.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
        total += len(inputs)
        # train_LOSS.append(running_loss)
        if i % 20 == 0:
            # print('Epoch ' + str(epoch) + ' : ' + str(i) + ' , LOSS =' + str(running_loss))
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch, i, running_loss / (i + 1), 100. * correct / total))
    lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
    scheduler.step()
    train_LOSS.append(running_loss / (i + 1))
    return lrs,train_LOSS
def model_valid_process(valid_loader,model,criterion):
    num_valid_examples,num_valid_los_examples,num_valid_nlos_examples = 0,0,0
    valid_loss, correct,correct_los,correct_nlos = 0, 0,0,0

    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.float(), labels.float()
        pred = model(inputs)
        # print('pred and label is',pred,labels)
        # valid_loss += len(inputs) * criterion(torch.squeeze(pred), labels)
        label_one = labels.argmax(1)
        los_idx = np.argwhere(label_one == 0)
        nlos_idx = np.argwhere(label_one == 1)
        los_pred = torch.squeeze(pred[los_idx])
        nlos_pred = torch.squeeze(pred[nlos_idx])
        los_label = torch.squeeze(labels[los_idx])
        nlos_label = torch.squeeze(labels[nlos_idx])
        # print('total lable',len(los_idx),len(nlos_idx))
        # print('label max', label_one)
        # print('los idx ',los_idx)
        # print('nlos idx',nlos_idx)
        # print('pred,',los_pred.argmax(1),pred.argmax(1))
        valid_loss += criterion(pred, torch.max(labels, 1)[1])
        correct += (pred.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
        num_valid_examples += len(inputs)
        correct_los += (los_pred.argmax(1) == los_label.argmax(1)).type(torch.float).sum().item()
        num_valid_los_examples += len(los_label)
        correct_nlos += (nlos_pred.argmax(1) == nlos_label.argmax(1)).type(torch.float).sum().item()
        num_valid_nlos_examples += len(nlos_label)
    valid_loss /= len(valid_loader)
    correct /= num_valid_examples
    correct_los /= num_valid_los_examples
    correct_nlos /= num_valid_nlos_examples
    return {
        "accuracy": correct,
        "accuracy_los":correct_los,
        "accuracy_nlos":correct_nlos,
        "loss": valid_loss
    }

set_random(0)
# In[3]:model para
in_channel=[2,4,8]
out_channel=[4,8,16]
num_clase = 2
model = Model_ericssion_one(num_clase, in_channel, out_channel)
print(model)
# for name, param in model.named_parameters():
#     print('Name:', name, 'Size:', param.size())

# data load
train_loader,valid_loader,test_loader = get_data_loader(r'./data_pos/data_cir_everysample_0412.pkl',batch_size=100)
print('data load return is:',len(train_loader),len(valid_loader),len(test_loader))

# in_tensor = torch.randn(batch_size, 2, 256, requires_grad=True)#ori
# print('in_rensor_size',in_tensor.size())
# gt_labels = torch.empty(batch_size).random_(2)
# print('lable size',gt_labels.shape,gt_labels)
#main func
epochs=11
#In[4]:loss function&optimizer
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),weight_decay=1e-2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs,eta_min=1e-6)
scheduler = WarmupCosineLR(optimizer, 5e-6, 2e-3,5, epochs, 0)
# scheduler = torch.optim.lr_scheduler.(optimizer,)
lrs = []
train_LOSS= []
# valid_LOSS=[]
valid_loss_total=[]
#In[5]:load checkpoint
start_epoch=30
try:
    checkpoint = torch.load(r'./save_model/412_10w_everysample_withoutzscore/0411_10w_cir_zscore_checkpoint_epoch_30.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

for epoch in range(epochs):
    print(f"Epoch {epoch}\n-------------------------------")
    model.train()
    lrs,train_LOSS= model_train_process(train_loader,model,optimizer,scheduler,criterion,train_LOSS,lrs)
    # print('lrs is:',len(lrs),lrs,'loss is ',len(LOSS),LOSS)
    model.eval()
    valid_eval_dict = model_valid_process(valid_loader,model,criterion)
    valid_accuracy = valid_eval_dict['accuracy'] * 100
    valid_loss = valid_eval_dict['loss']
    valid_loss_total.append(valid_loss.detach().numpy())
    print(f"Test Error: \n Accuracy: {valid_accuracy:>0.1f}%, Avg loss: {valid_loss:>8f}\n")
    # save model
    if epoch % 30 == 0 and epoch !=0:
        torch.save({
            'epoch': epoch+start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        },r'./save_model/412_10w_everysample_withoutzscore/0411_10w_cir_zscore_checkpoint_epoch_{}.tar'.format(epoch+start_epoch))
    if epoch == epochs-1:
        # torch.save(model, r'./save_model/0327_update_normalize_total_epoch_'+str(epoch)+'.pth')
        # np.save(r'./save_model/2_class_lrsinf_epoch_{}'.format(epoch), lrs)
        np.save(r'./save_model/412_10w_everysample_withoutzscore/0411_10w_cir_zscore_trainLOSS_epoch_{}'.format(epoch+start_epoch), train_LOSS)
        np.save(r'./save_model/412_10w_everysample_withoutzscore/0411_10w_cir_zscore_validLOSS_epoch_{}'.format(epoch+start_epoch), valid_loss_total)


#test
model.eval()
test_eval_dict = model_valid_process(test_loader, model, criterion)
test_accuracy = test_eval_dict['accuracy'] * 100
test_loss = test_eval_dict['loss']
test_los_acc = test_eval_dict['accuracy_los']*100
test_nlos_acc = test_eval_dict['accuracy_nlos']*100
print(f"Test Error: \n Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f}\n accuracy_los:{test_los_acc:>0.1f}%,accuracy_nlos:{test_nlos_acc:>0.1f}\n")
# print('len loss',len(train_LOSS),len(valid_LOSS_))
#画图
plt.figure(figsize=(10, 6))
plt.plot(lrs, color='r')
plt.text(0, lrs[0], str(lrs[0]))
plt.text(epochs, lrs[-1], str(lrs[-1]))
plt.figure(figsize=(10, 6))
plt.plot(train_LOSS[2:],color='b')
plt.plot(valid_loss_total[2:],color='y')

plt.xlabel('epochs')
plt.ylabel('LOSS')
plt.title('epoch:{}'.format(start_epoch)+'-{}'.format(start_epoch+epochs))
plt.legend(['train_loss','valid_loss'])
plt.savefig('./save_model/412_10w_everysample_withoutzscore/411_fig_epoch_{}'.format(start_epoch)+'_{}.png'.format(start_epoch+epochs))
plt.show()