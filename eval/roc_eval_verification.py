import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import os, sys, copy
sys.path.insert(0, os.path.abspath('.'))
from data.data_loader import ConvertRGB2BGR, FixedImageStandard
from network import load_model
from configs import datasets_config as config
import network.fsb_hash_net as net
import network.fsb_hash_net_baseline as net_base

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 500
eer_dict = {'ethnic' : 0, 'pubfig' : 0, 'facescrub': 0, 'imdb_wiki' : 0, 'ar' : 0}
eer_dict = {}
auc_dict = {}
fpr_dict = {}
tpr_dict = {}
dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
ver_img_per_class = 4


def get_avg(dict_list):
    total_eer = 0
    eer_list = []
    if 'avg' in dict_list.keys():
        del dict_list['avg']
    if 'std' in dict_list.keys():
        del dict_list['std']
    for items in dict_list:
        total_eer += dict_list[items]
        eer_list.append(dict_list[items])
    dict_list['avg'] = total_eer/len(dict_list) * 100
    dict_list['std'] = np.std(np.array(eer_list)) * 100

    return dict_list


def create_folder(method):
    lists = ['peri', 'face', 'cm']
    boiler_path = './data/roc/'
    for modal in lists:
        if not os.path.exists(os.path.join(boiler_path, method, modal)):
            os.makedirs(os.path.join(boiler_path, method, modal))


def compute_eer(fpr,tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    eer = np.around(eer, 4)
    return eer


class dataset(data.Dataset):
    def __init__(self, dset, root_drt, modal, dset_type='gallery'):
        if modal[:4] == 'peri':
            sz = (112, 112)
        elif modal[:4] == 'face':
            sz = (112, 112)
        
        self.ocular_root_dir = os.path.join(os.path.join(root_drt, dset, dset_type), modal)
        self.nof_identity = len(os.listdir(self.ocular_root_dir))
        self.ocular_img_dir_list = []
        self.label_list = []
        self.label_dict = {}
        cnt = 0
        for iden in sorted(os.listdir(self.ocular_root_dir)):
            ver_img_cnt = 0
            for i in range(ver_img_per_class):
                list_img = sorted(os.listdir(os.path.join(self.ocular_root_dir, iden)))
                list_len = len(list_img)
                offset = list_len // ver_img_per_class
                self.ocular_img_dir_list.append(os.path.join(self.ocular_root_dir, iden, list_img[offset*i]))
                self.label_list.append(cnt)
                ver_img_cnt += 1
                if ver_img_cnt == ver_img_per_class:
                    break
            cnt += 1

        self.onehot_label = np.zeros((len(self.ocular_img_dir_list), self.nof_identity))
        for i in range(len(self.ocular_img_dir_list)):
            self.onehot_label[i, self.label_list[i]] = 1

        self.ocular_transform = transforms.Compose([transforms.Resize(sz),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    def __len__(self):
        return len(self.ocular_img_dir_list)

    def __getitem__(self, idx):
        ocular = Image.open(self.ocular_img_dir_list[idx])
        ocular = self.ocular_transform(ocular)
        onehot = self.onehot_label[idx]
        lbl = self.label_list[idx]
        return ocular, onehot, lbl
    
#### Validate verification (Intra-Modal)
def val_verify(feature_extractor, generator, dset_name, emb_size = 1024, peri_flag=False, root_drt=config.evaluation['verification'], device='cuda:0', mode='user'):
    modal = 'peri' if peri_flag == True else 'face'

    embedding_size = emb_size       
    
    dset = dataset(dset=dset_name, dset_type='val', root_drt = root_drt, modal=modal)

    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    nof_dset = len(dset)
    nof_iden = dset.nof_identity
    embedding_mat = torch.zeros((nof_dset, embedding_size)).to(device)
    label_mat = torch.zeros((nof_dset, nof_iden)).to(device)

    feature_extractor = feature_extractor.eval().to(device)
    generator = generator.eval().to(device)

    with torch.no_grad():
        for i, (ocular, onehot, lbl) in enumerate(dloader):
            nof_img = ocular.shape[0]
            ocular = ocular.to(device)
            onehot = onehot.to(device)
            lbl = lbl.to(device)

            if mode == 'stolen':
                lbl = torch.zeros_like(lbl)

            feature = feature_extractor(ocular)
            hash = generator(feature, lbl)

            embedding_mat[i*batch_size:i*batch_size+nof_img, :] = hash.detach().clone()                
            label_mat[i*batch_size:i*batch_size+nof_img, :] = onehot

        ### roc
        embedding_mat /= torch.norm(embedding_mat, p=2, dim=1, keepdim=True)

        score_mat = torch.matmul(embedding_mat, embedding_mat.t()).cpu()
        gen_mat = torch.matmul(label_mat, label_mat.t()).cpu()
        gen_r, gen_c = torch.where(gen_mat == 1)
        imp_r, imp_c = torch.where(gen_mat == 0)

        gen_score = score_mat[gen_r, gen_c].cpu().numpy()
        imp_score = score_mat[imp_r, imp_c].cpu().numpy()

        y_gen = np.ones(gen_score.shape[0])
        y_imp = np.zeros(imp_score.shape[0])

        score = np.concatenate((gen_score, imp_score))
        y = np.concatenate((y_gen, y_imp))

        fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
        eer_value = compute_eer(fpr_tmp, tpr_tmp)

    return eer_value
    
    
#### Intra Modal Verification
def im_verify(feature_extractor, generator, emb_size = 1024, peri_flag=False,  root_drt=config.evaluation['verification'], device='cuda:0', eval_mode='verify', mode='user'):
    modal = 'peri' if peri_flag == True else 'face'

    for dset_name in dset_list:
        embedding_size = emb_size       
        
        if dset_name == 'ethnic':
            dset = dataset(dset=dset_name, dset_type='Verification/gallery', root_drt = root_drt, modal=modal)
        else:
            dset = dataset(dset=dset_name, dset_type='gallery', root_drt = root_drt, modal=modal)

        dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
        nof_dset = len(dset)
        nof_iden = dset.nof_identity
        embedding_mat = torch.zeros((nof_dset, embedding_size)).to(device)
        label_mat = torch.zeros((nof_dset, nof_iden)).to(device)

        feature_extractor = feature_extractor.eval().to(device)
        generator = generator.eval().to(device)

        with torch.no_grad():
            for i, (ocular, onehot, lbl) in enumerate(dloader):
                nof_img = ocular.shape[0]
                ocular = ocular.to(device)
                onehot = onehot.to(device)
                lbl = lbl.to(device)

                if mode == 'stolen':
                    lbl = torch.zeros_like(lbl)

                feature = feature_extractor(ocular)
                hash = generator(feature, lbl)

                embedding_mat[i*batch_size:i*batch_size+nof_img, :] = hash.detach().clone()                
                label_mat[i*batch_size:i*batch_size+nof_img, :] = onehot

            ### roc
            embedding_mat /= torch.norm(embedding_mat, p=2, dim=1, keepdim=True)

            score_mat = torch.matmul(embedding_mat, embedding_mat.t()).cpu()
            gen_mat = torch.matmul(label_mat, label_mat.t()).cpu()
            gen_r, gen_c = torch.where(gen_mat == 1)
            imp_r, imp_c = torch.where(gen_mat == 0)

            gen_score = score_mat[gen_r, gen_c].cpu().numpy()
            imp_score = score_mat[imp_r, imp_c].cpu().numpy()

            y_gen = np.ones(gen_score.shape[0])
            y_imp = np.zeros(imp_score.shape[0])

            score = np.concatenate((gen_score, imp_score))
            y = np.concatenate((y_gen, y_imp))

            fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
            auc = roc_auc_score(y, score)
            fpr_dict[dset_name] = fpr_tmp
            tpr_dict[dset_name] = tpr_tmp
            auc_dict[dset_name] = auc
            eer_dict[dset_name] = compute_eer(fpr_tmp, tpr_tmp)

    if eval_mode == 'verify':
        return eer_dict
    elif eval_mode == 'roc':
        return eer_dict, fpr_dict, tpr_dict, auc_dict


#### Cross-Modal Verification
def cm_verify(feature_extractor, generator, emb_size=1024, root_drt=config.evaluation['verification'], device='cuda:0', eval_mode='verify', mode='user'):
    for dset_name in dset_list:
        embedding_size = emb_size       
        
        if dset_name == 'ethnic':
            peri_dset = dataset(dset=dset_name, dset_type='Verification/gallery', root_drt = root_drt, modal='peri')
            face_dset = dataset(dset=dset_name, dset_type='Verification/gallery', root_drt = root_drt, modal='face')
        else:
            peri_dset = dataset(dset=dset_name, dset_type='gallery', root_drt = root_drt, modal='peri')
            face_dset = dataset(dset=dset_name, dset_type='gallery', root_drt = root_drt, modal='face')

        peri_dloader = torch.utils.data.DataLoader(peri_dset, batch_size=batch_size, num_workers=4)
        nof_peri_dset = len(peri_dset)
        nof_peri_iden = peri_dset.nof_identity
        peri_embedding_mat = torch.zeros((nof_peri_dset, embedding_size)).to(device)
        peri_label_mat = torch.zeros((nof_peri_dset, nof_peri_iden)).to(device)

        face_dloader = torch.utils.data.DataLoader(face_dset, batch_size=batch_size, num_workers=4)
        nof_face_dset = len(face_dset)
        nof_face_iden = face_dset.nof_identity
        face_embedding_mat = torch.zeros((nof_face_dset, embedding_size)).to(device)
        face_label_mat = torch.zeros((nof_face_dset, nof_face_iden)).to(device)

        label_mat = torch.tensor([]).to(device)  

        feature_extractor = feature_extractor.eval().to(device)
        generator = generator.eval().to(device)

        with torch.no_grad():
            for i, (peri_ocular, peri_onehot, peri_lbl) in enumerate(peri_dloader):
                nof_peri_img = peri_ocular.shape[0]
                peri_ocular = peri_ocular.to(device)
                peri_onehot = peri_onehot.to(device)
                peri_lbl = peri_lbl.to(device)

                if mode == 'stolen':
                    peri_lbl = torch.zeros_like(peri_lbl)

                peri_feature = feature_extractor(peri_ocular)
                peri_hash = generator(peri_feature, peri_lbl)

                peri_embedding_mat[i*batch_size:i*batch_size+nof_peri_img, :] = peri_hash.detach().clone()                
                peri_label_mat[i*batch_size:i*batch_size+nof_peri_img, :] = peri_onehot


            for i, (face_ocular, face_onehot, face_lbl) in enumerate(face_dloader):
                nof_face_img = face_ocular.shape[0]
                face_ocular = face_ocular.to(device)
                face_onehot = face_onehot.to(device)
                face_lbl = face_lbl.to(device)

                if mode == 'stolen':
                    face_lbl = torch.zeros_like(face_lbl)

                face_feature = feature_extractor(face_ocular)
                face_hash = generator(face_feature, face_lbl)

                face_embedding_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_hash.detach().clone()
                face_label_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_onehot

            ### roc
            face_embedding_mat /= torch.norm(face_embedding_mat, p=2, dim=1, keepdim=True)
            peri_embedding_mat /= torch.norm(peri_embedding_mat, p=2, dim=1, keepdim=True)

            score_mat = torch.matmul(face_embedding_mat, peri_embedding_mat.t()).cpu()
            gen_mat = torch.matmul(face_label_mat, peri_label_mat.t()).cpu()
            gen_r, gen_c = torch.where(gen_mat == 1)
            imp_r, imp_c = torch.where(gen_mat == 0)

            gen_score = score_mat[gen_r, gen_c].cpu().numpy()
            imp_score = score_mat[imp_r, imp_c].cpu().numpy()

            y_gen = np.ones(gen_score.shape[0])
            y_imp = np.zeros(imp_score.shape[0])

            score = np.concatenate((gen_score, imp_score))
            y = np.concatenate((y_gen, y_imp))

            fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
            auc = roc_auc_score(y, score)
            fpr_dict[dset_name] = fpr_tmp
            tpr_dict[dset_name] = tpr_tmp
            auc_dict[dset_name] = auc
            eer_dict[dset_name] = compute_eer(fpr_tmp, tpr_tmp)

    if eval_mode == 'verify':
        return eer_dict
    elif eval_mode == 'roc':
        return eer_dict, fpr_dict, tpr_dict, auc_dict

if __name__ == '__main__':
    method = 'fsb_hashnet'
    eval_mode = 'roc'
    if eval_mode == 'roc':
        create_folder(method + '/nohash')
        create_folder(method + '/stolen')
        create_folder(method + '/user')
    embd_dim = 512
    device = torch.device('cuda:0')

    # Baseline Model (No Hash)
    load_model_path_fe_base = './models/best_feature_extractor/FSB-HashNet_Baseline.pth'
    feature_extractor_base = net_base.FSB_Hash_Net(embedding_size = 1024, do_prob = 0.0).eval().to(device)
    feature_extractor_base = load_model.load_pretrained_network(feature_extractor_base, load_model_path_fe_base, device = device)
    
    load_model_path_base = load_model_path_fe_base.replace('best_feature_extractor', 'best_generator')
    generator_base = net_base.Hash_Generator(embedding_size = 1024, device=device, out_embedding_size=512).eval().to(device)
    generator_base = load_model.load_pretrained_network(generator_base, load_model_path_base, device = device)

    # FSB-HashNet Model
    load_model_path_fe = './models/best_feature_extractor/FSB-HashNet.pth'
    feature_extractor = net.FSB_Hash_Net(embedding_size = 1024, do_prob = 0.0).eval().to(device)
    feature_extractor = load_model.load_pretrained_network(feature_extractor, load_model_path_fe, device = device)
    
    load_model_path = load_model_path_fe.replace('best_feature_extractor', 'best_generator')
    generator = net.Hash_Generator(embedding_size = 1024, device=device, out_embedding_size=512).eval().to(device)
    generator = load_model.load_pretrained_network(generator, load_model_path, device = device)


    #### No Hash (Baseline Model)
    print('No Hash \n')
    nohash_peri_eer_dict, nohash_peri_fpr_dict, nohash_peri_tpr_dict, nohash_peri_auc_dict = im_verify(feature_extractor_base, generator_base, embd_dim, root_drt=config.evaluation['verification'], peri_flag=True, device=device, eval_mode=eval_mode, mode='nohash')
    nohash_peri_eer_dict = get_avg(nohash_peri_eer_dict)
    if eval_mode == 'roc':
        nohash_peri_eer_dict = copy.deepcopy(nohash_peri_eer_dict)
        nohash_peri_fpr_dict = copy.deepcopy(nohash_peri_fpr_dict)
        nohash_peri_tpr_dict = copy.deepcopy(nohash_peri_tpr_dict)
        nohash_peri_auc_dict = copy.deepcopy(nohash_peri_auc_dict)    
        torch.save(nohash_peri_eer_dict, './data/roc/' + str(method) + '/nohash/peri/peri_eer_dict.pt')
        torch.save(nohash_peri_fpr_dict, './data/roc/' + str(method) + '/nohash/peri/peri_fpr_dict.pt')
        torch.save(nohash_peri_tpr_dict, './data/roc/' + str(method) + '/nohash/peri/peri_tpr_dict.pt')
        torch.save(nohash_peri_auc_dict, './data/roc/' + str(method) + '/nohash/peri/peri_auc_dict.pt')
    print('Average EER (Intra-Modal Periocular):', nohash_peri_eer_dict['avg'], '±', nohash_peri_eer_dict['std'])

    nohash_face_eer_dict, nohash_face_fpr_dict, nohash_face_tpr_dict, nohash_face_auc_dict = im_verify(feature_extractor_base, generator_base, embd_dim, root_drt=config.evaluation['verification'], peri_flag=False, device=device, eval_mode=eval_mode, mode='nohash')
    nohash_face_eer_dict = get_avg(nohash_face_eer_dict)
    if eval_mode == 'roc':
        nohash_face_eer_dict = copy.deepcopy(nohash_face_eer_dict)
        nohash_face_fpr_dict = copy.deepcopy(nohash_face_fpr_dict)
        nohash_face_tpr_dict = copy.deepcopy(nohash_face_tpr_dict)
        nohash_face_auc_dict = copy.deepcopy(nohash_face_auc_dict)    
        torch.save(nohash_face_eer_dict, './data/roc/' + str(method) + '/nohash/face/face_eer_dict.pt')
        torch.save(nohash_face_fpr_dict, './data/roc/' + str(method) + '/nohash/face/face_fpr_dict.pt')
        torch.save(nohash_face_tpr_dict, './data/roc/' + str(method) + '/nohash/face/face_tpr_dict.pt')
        torch.save(nohash_face_auc_dict, './data/roc/' + str(method) + '/nohash/face/face_auc_dict.pt')
    print('Average EER (Intra-Modal Face):', nohash_face_eer_dict['avg'], '±', nohash_face_eer_dict['std'])

    nohash_cm_eer_dict, nohash_cm_fpr_dict, nohash_cm_tpr_dict, nohash_cm_auc_dict = cm_verify(feature_extractor_base, generator_base, emb_size=embd_dim, root_drt=config.evaluation['verification'], device=device, eval_mode=eval_mode, mode='nohash')
    if eval_mode == 'roc':
        nohash_cm_eer_dict = get_avg(nohash_cm_eer_dict) 
        nohash_cm_eer_dict = copy.deepcopy(nohash_cm_eer_dict)
        nohash_cm_fpr_dict = copy.deepcopy(nohash_cm_fpr_dict)
        nohash_cm_tpr_dict = copy.deepcopy(nohash_cm_tpr_dict)
        nohash_cm_auc_dict = copy.deepcopy(nohash_cm_auc_dict)    
        torch.save(nohash_cm_eer_dict, './data/roc/' + str(method) + '/nohash/cm/cm_eer_dict.pt')
        torch.save(nohash_cm_fpr_dict, './data/roc/' + str(method) + '/nohash/cm/cm_fpr_dict.pt')
        torch.save(nohash_cm_tpr_dict, './data/roc/' + str(method) + '/nohash/cm/cm_tpr_dict.pt')
        torch.save(nohash_cm_auc_dict, './data/roc/' + str(method) + '/nohash/cm/cm_auc_dict.pt')
    print('Average EER (Cross-Modal):', nohash_cm_eer_dict['avg'], '±', nohash_cm_eer_dict['std'])

    
    #### Stolen Token Scenario
    print('Stolen Token Scenario \n')
    stolen_peri_eer_dict, stolen_peri_fpr_dict, stolen_peri_tpr_dict, stolen_peri_auc_dict = im_verify(feature_extractor, generator, embd_dim, root_drt=config.evaluation['verification'], peri_flag=True, device=device, eval_mode=eval_mode, mode='stolen')
    stolen_peri_eer_dict = get_avg(stolen_peri_eer_dict)
    if eval_mode == 'roc':
        stolen_peri_eer_dict = copy.deepcopy(stolen_peri_eer_dict)
        stolen_peri_fpr_dict = copy.deepcopy(stolen_peri_fpr_dict)
        stolen_peri_tpr_dict = copy.deepcopy(stolen_peri_tpr_dict)
        stolen_peri_auc_dict = copy.deepcopy(stolen_peri_auc_dict)    
        torch.save(stolen_peri_eer_dict, './data/roc/' + str(method) + '/stolen/peri/peri_eer_dict.pt')
        torch.save(stolen_peri_fpr_dict, './data/roc/' + str(method) + '/stolen/peri/peri_fpr_dict.pt')
        torch.save(stolen_peri_tpr_dict, './data/roc/' + str(method) + '/stolen/peri/peri_tpr_dict.pt')
        torch.save(stolen_peri_auc_dict, './data/roc/' + str(method) + '/stolen/peri/peri_auc_dict.pt')
    print('Average EER (Intra-Modal Periocular):', stolen_peri_eer_dict['avg'], '±', stolen_peri_eer_dict['std'])

    stolen_face_eer_dict, stolen_face_fpr_dict, stolen_face_tpr_dict, stolen_face_auc_dict = im_verify(feature_extractor, generator, embd_dim, root_drt=config.evaluation['verification'], peri_flag=False, device=device, eval_mode=eval_mode, mode='stolen')
    stolen_face_eer_dict = get_avg(stolen_face_eer_dict)
    if eval_mode == 'roc':
        stolen_face_eer_dict = copy.deepcopy(stolen_face_eer_dict)
        stolen_face_fpr_dict = copy.deepcopy(stolen_face_fpr_dict)
        stolen_face_tpr_dict = copy.deepcopy(stolen_face_tpr_dict)
        stolen_face_auc_dict = copy.deepcopy(stolen_face_auc_dict)    
        torch.save(stolen_face_eer_dict, './data/roc/' + str(method) + '/stolen/face/face_eer_dict.pt')
        torch.save(stolen_face_fpr_dict, './data/roc/' + str(method) + '/stolen/face/face_fpr_dict.pt')
        torch.save(stolen_face_tpr_dict, './data/roc/' + str(method) + '/stolen/face/face_tpr_dict.pt')
        torch.save(stolen_face_auc_dict, './data/roc/' + str(method) + '/stolen/face/face_auc_dict.pt')
    print('Average EER (Intra-Modal Face):', stolen_face_eer_dict['avg'], '±', stolen_face_eer_dict['std'])

    stolen_cm_eer_dict, stolen_cm_fpr_dict, stolen_cm_tpr_dict, stolen_cm_auc_dict = cm_verify(feature_extractor, generator, emb_size=embd_dim, root_drt=config.evaluation['verification'], device=device, eval_mode=eval_mode, mode='stolen')
    if eval_mode == 'roc':
        stolen_cm_eer_dict = get_avg(stolen_cm_eer_dict) 
        stolen_cm_eer_dict = copy.deepcopy(stolen_cm_eer_dict)
        stolen_cm_fpr_dict = copy.deepcopy(stolen_cm_fpr_dict)
        stolen_cm_tpr_dict = copy.deepcopy(stolen_cm_tpr_dict)
        stolen_cm_auc_dict = copy.deepcopy(stolen_cm_auc_dict)    
        torch.save(stolen_cm_eer_dict, './data/roc/' + str(method) + '/stolen/cm/cm_eer_dict.pt')
        torch.save(stolen_cm_fpr_dict, './data/roc/' + str(method) + '/stolen/cm/cm_fpr_dict.pt')
        torch.save(stolen_cm_tpr_dict, './data/roc/' + str(method) + '/stolen/cm/cm_tpr_dict.pt')
        torch.save(stolen_cm_auc_dict, './data/roc/' + str(method) + '/stolen/cm/cm_auc_dict.pt')
    print('Average EER (Cross-Modal):', stolen_cm_eer_dict['avg'], '±', stolen_cm_eer_dict['std'])


    #### User-Specific Token Scenario
    print('User-Specific Token Scenario \n')
    user_peri_eer_dict, user_peri_fpr_dict, user_peri_tpr_dict, user_peri_auc_dict = im_verify(feature_extractor, generator, embd_dim, root_drt=config.evaluation['verification'], peri_flag=True, device=device, eval_mode=eval_mode, mode='user')
    if eval_mode == 'roc':
        user_peri_eer_dict = get_avg(user_peri_eer_dict) 
        user_peri_eer_dict = copy.deepcopy(user_peri_eer_dict)
        user_peri_fpr_dict = copy.deepcopy(user_peri_fpr_dict)
        user_peri_tpr_dict = copy.deepcopy(user_peri_tpr_dict)
        user_peri_auc_dict = copy.deepcopy(user_peri_auc_dict)    
        torch.save(user_peri_eer_dict, './data/roc/' + str(method) + '/user/peri/peri_eer_dict.pt')
        torch.save(user_peri_fpr_dict, './data/roc/' + str(method) + '/user/peri/peri_fpr_dict.pt')
        torch.save(user_peri_tpr_dict, './data/roc/' + str(method) + '/user/peri/peri_tpr_dict.pt')
        torch.save(user_peri_auc_dict, './data/roc/' + str(method) + '/user/peri/peri_auc_dict.pt')
    print('Average EER (Intra-Modal Periocular):', user_peri_eer_dict['avg'], '±', user_peri_eer_dict['std'])

    user_face_eer_dict, user_face_fpr_dict, user_face_tpr_dict, user_face_auc_dict = im_verify(feature_extractor, generator, embd_dim, root_drt=config.evaluation['verification'], peri_flag=False, device=device, eval_mode=eval_mode, mode='user')
    if eval_mode == 'roc':
        user_face_eer_dict = get_avg(user_face_eer_dict) 
        user_face_eer_dict = copy.deepcopy(user_face_eer_dict)
        user_face_fpr_dict = copy.deepcopy(user_face_fpr_dict)
        user_face_tpr_dict = copy.deepcopy(user_face_tpr_dict)
        user_face_auc_dict = copy.deepcopy(user_face_auc_dict)    
        torch.save(user_face_eer_dict, './data/roc/' + str(method) + '/user/face/face_eer_dict.pt')
        torch.save(user_face_fpr_dict, './data/roc/' + str(method) + '/user/face/face_fpr_dict.pt')
        torch.save(user_face_tpr_dict, './data/roc/' + str(method) + '/user/face/face_tpr_dict.pt')
        torch.save(user_face_auc_dict, './data/roc/' + str(method) + '/user/face/face_auc_dict.pt')
    print('Average EER (Intra-Modal Face):', user_face_eer_dict['avg'], '±', user_face_eer_dict['std'])

    user_cm_eer_dict, user_cm_fpr_dict, user_cm_tpr_dict, user_cm_auc_dict = cm_verify(feature_extractor, generator, emb_size=embd_dim, root_drt=config.evaluation['verification'], device=device, eval_mode=eval_mode, mode='user')    
    if eval_mode == 'roc':
        user_cm_eer_dict = get_avg(user_cm_eer_dict) 
        user_cm_eer_dict = copy.deepcopy(user_cm_eer_dict)
        user_cm_fpr_dict = copy.deepcopy(user_cm_fpr_dict)
        user_cm_tpr_dict = copy.deepcopy(user_cm_tpr_dict)
        user_cm_auc_dict = copy.deepcopy(user_cm_auc_dict)    
        torch.save(user_cm_eer_dict, './data/roc/' + str(method) + '/user/cm/cm_eer_dict.pt')
        torch.save(user_cm_fpr_dict, './data/roc/' + str(method) + '/user/cm/cm_fpr_dict.pt')
        torch.save(user_cm_tpr_dict, './data/roc/' + str(method) + '/user/cm/cm_tpr_dict.pt')
        torch.save(user_cm_auc_dict, './data/roc/' + str(method) + '/user/cm/cm_auc_dict.pt')
    print('Average EER (Cross-Modal):', user_cm_eer_dict['avg'], '±', user_cm_eer_dict['std'])