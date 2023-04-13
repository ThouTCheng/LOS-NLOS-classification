import os
import pickle
from  sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import h5py
# path = r'D:\2023-work\task\AI&machine learning\xcector_LOS_Identification\时域信道估计结果/LOS.mat'
# los_CIR = scio.loadmat(path)
# # print(type(los_CIR))
# #print(los_CIR['LOS'])
# #print(len(los_CIR['H_time_domain']))
# print(los_CIR['H_time_domain'].shape)
# # for item in los_CIR.items():
# #     print(item[0])
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
def gen_train_data(data,label,path,clases,train_sample_num):
    # _min, _max = float('inf'), -float('inf')
    cir_line = 2
    cir_len = 256
    for file_name in os.listdir(path):
        #print('file name is',file_name)
        if file_name.endswith('mat'):
            print('file name is',file_name)
            CIR = scio.loadmat(path+r'/'+file_name)
            CIR_data = CIR['H_time_domain']#array(1000*256)
            CIR_data_douple=list(zip(CIR_data.real,CIR_data.imag))#list(10000(tuple(2*256)))
            #print(CIR_data[0:5,0:10])
            # aa=np.array(CIR_data_douple[0])
            # print(aa[:,0:5])
            #break
            _min = np.amin(CIR_data_douple)
            _max = np.amax(CIR_data_douple)
            # print('max and min is ',_max,_min)
            #print(_max,_min)
            for i in range(train_sample_num):
                if np.array(CIR_data_douple[i]).shape[0] == 2 and np.array(CIR_data_douple[i]).shape[1] == 256:
                    isnan = np.isnan(np.array(CIR_data_douple[i]))
                    if True in isnan:
                        print("我有nan啊！{}".format(i))
                    else:
                        stand_data = preprocessing.normalize(np.array(CIR_data_douple[i]),norm='l2')
                        # data.append(np.array(CIR_data_douple[i]))#data[0]'s shape is 2*256
                        data.append(stand_data)
                        label.append(to_categorical(clases.index(file_name.split('.mat')[0]), len(clases)))
                    # 每条数据归一化
                    # tmp_data =np.array(CIR_data_douple[i])
                    # max_tmp = np.amax(tmp_data)
                    # min_tmp = np.amin(tmp_data)
                    # tmp_data = (tmp_data-min_tmp)/(max_tmp-min_tmp)
                    # data.append(tmp_data)
                    # label.append(clases.index(file_name.split('.mat')[0]))

    #归一化
    # data = (data - _min) / (_max - _min)
    # data = data/_max#所有数据一起归一化
    #均值方差归一化
    data = np.array(data,dtype=float)
    mean_all = np.mean(data,axis=0)
    std_all = np.std(data,axis=0)
    data_norm = (data-mean_all)/std_all
    # print('mean shape,',mean_all.shape,std_all.shape)
    # print('mean and std is',mean_all,'\n',std_all)
    #plt
    # print('data is ', data[0].shape,data[0][1,:].shape,np.amax(data[13116]))
    # # print('data ori',np.argmax(data[20008][0, :] ** 2 + data[20008][1,:] ** 2))
    # plt.plot(data_norm[121][0, :] ** 2 + data_norm[121][1,:] ** 2)
    # plt.plot(data_norm[14220][0, :] ** 2 + data_norm[14220][1,:] ** 2)
    # # # print('[][]',np.amax(data[710][0, :]**2),np.amax(data[710][1,:]**2))
    # # print('[][][]',np.amax( data[710][0, :] ** 2 + data[710][1,:] ** 2))
    # plt.show()
    # # #
    print('len train data is:',len(data))
    # print('len train data_norm is:',len(data_norm))
    print('size of tran sample is :',data[0].shape,data[0][:,1:10])
    # print('size of tran sample is :',data_norm[0].shape)
    # #print(data[1])
    # print('label len is',len(label))
    with open(r'./data_pos/data_cir_everysample_0412.pkl','wb') as f:
         pickle.dump((data_norm,label),f)
    print('data wirte done!')



def read_toa_data(path):
    data=[]
    label=[]
    for file_name in os.listdir(path):
        if file_name.endswith('mat'):
            print('file name is', file_name)
            # total_data = scio.loadmat(path + r'/' + file_name)
            total_data = h5py.File(path + r'/' + file_name)
            # print('[][][] mat data type',type(total_data),total_data.keys())
            CIR_data = total_data['H']#(30000, 18, 2, 256)
            TOA_label = total_data['TOA']#(18, 30000)
            # print('[][][] CIR data shape', CIR_data[0,0,:,:])
            # print('[][][] toa data shape', TOA_label.shape, '\n', TOA_label[:, 5])
            print('[][][] CIR_data shape',CIR_data.shape[0])

            num_sample = CIR_data.shape[0]
            num_TRP = CIR_data.shape[1]
            for i in range(num_sample):
                for j in range(7):
                    if np.array(CIR_data[i,j,:,:]).shape[0] == 2 and np.array(CIR_data[i,j,:,:]).shape[1] == 256:
                        # print('example data:',np.array(CIR_data[i,j,:,:]),np.array(CIR_data[i,j,:,:].shape))
                        X_train=np.array(CIR_data[i,j,:,:])
                        isnan = np.isnan(X_train)
                        if True in isnan:
                            print("我有nan啊！{}".format(i))
                        else:
                            data.append(X_train)
                            label.append(TOA_label[j,i])
    print('[][][] len data',len(data),data[0].shape,type(data))
    print('[][][] len label',len(label),max(label),min(label))
    print('[][][]',data[0][:,1:5])
    #标准化
    # stand_data = preprocessing.StandardScaler().fit_transform(data)
    # print(data[20])
    # plt.plot(data[20][0,:] ** 2 + data[20][1,:] ** 2)
    # plt.show()
    #Zscore归一化
    data = np.array(data, dtype=float)
    # print('type of data',type(data),data.shape)
    mean_all = np.mean(data, axis=0)
    std_all = np.std(data, axis=0)
    isnan = np.where(std_all==0)
    print("我有0啊！",isnan)
    print('mean and std shape,', mean_all.shape, std_all.shape)

    # print('[][]std all is',std_all)
    data_norm = (data - mean_all) / std_all
    data_norm[:,:,0:2]=0 #256的前两维是0
    # print('mean and std is','\n',std_all,mean_all)
    print('[][][] len data', len(data_norm), data_norm[0].shape, type(data_norm))
    # print('[][][]', data_norm[0][:, 0:5])
    #save data
    with open(r'./data_toa/412_TOA_data_pos2_Zscore_21w.pkl','wb') as f:
         pickle.dump((data_norm,label),f)
    print('data wirte done!')

def read_toa_Data_form_zsl(path):
    data=[]
    label=[]
    for file_name in os.listdir(path):
        if file_name.endswith('mat'):
            print('file name is', file_name)
            total_data = scio.loadmat(path + r'/' + file_name)
            CIR_data = total_data['H_time_domain']#(50000*256)
            print('len CIR data is ',len(CIR_data))
            TOA_label = np.squeeze(total_data['ToA'])#(1*50000)
            CIR_data_douple = list(zip(CIR_data.real, CIR_data.imag))
            # print('len data and label is',len(CIR_data_douple),TOA_label.shape)
            for i in range(len(CIR_data)):
                if np.array(CIR_data_douple[i]).shape[0] == 2 and np.array(CIR_data_douple[i]).shape[1] == 256:
                    isnan = np.isnan(np.array(CIR_data_douple[i]))
                    if True in isnan:
                        print("我有nan啊！{}".format(i))
                    else:
                        # stand_data = preprocessing.normalize(np.array(CIR_data_douple[i]), norm='l2')
                        # data.append(np.array(CIR_data_douple[i]))#data[0]'s shape is 2*256
                        x_data = np.array(CIR_data_douple[i])
                        data.append(x_data)
                        label.append(TOA_label[i]*3e8)
    #Zscore 归一化
    data = np.array(data,dtype=float)
    mean_all = np.mean(data,axis=0)
    std_all = np.std(data,axis=0)
    data_norm = (data-mean_all)/std_all
    print('len data and lable is',len(data),len(label))
    print('len mean and std is',mean_all.shape,std_all.shape)
    print('len data and lable is', len(data_norm),'example data is \n',data_norm[0][0,0:10])
    print('example label is',label[1:10])
    #write data
    with open(r'./data_toa/412_zsl_5w.pkl','wb') as f:
        pickle.dump((data_norm, label), f)
    print('data wirte done!')










if __name__ == '__main__':
    # data=[]
    # label=[]
    # #读取zsl给的mat数据，并生成可用于训练的pkl文件
    # path=r'D:\2023-work\task\AI&machine learning\xcector_LOS_Identification\AI results 04-11'
    # clases=['Inf_LOS','Inf_NLOS']
    # # clases = ['LOS', 'NLOS']
    # train_sample_num = 50000
    # gen_train_data(data,label,path,clases,train_sample_num)

    # 读取信通院的open数据集中给的mat数据，并生成可用于训练的pkl文件
    # path_TOA = r'D:\2023-work\task\AI&machine learning\xcector_LOS_Identification\AI results 04-12'
    # read_toa_data(path_TOA)

    # 读取zsl给的toa文件
    path_TOA = r'D:\2023-work\task\AI&machine learning\xcector_LOS_Identification\AI results 04-12'
    read_toa_Data_form_zsl(path_TOA)

