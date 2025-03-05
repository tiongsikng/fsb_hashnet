import os, sys, copy
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from sklearn.metrics import roc_curve
sys.path.insert(0, os.path.abspath('.'))
import network.fsb_hash_net as net1
import network.fsb_hash_net_baseline as net2
from network import load_model
from configs import datasets_config as config
from torch.distributions import Beta

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 200
eer_dict = {}
ver_img_per_class = 250 # calculate as many possible samples


def compute_eer(fpr,tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    eer = np.around(eer, 4)
    return eer


class dataset(data.Dataset):
    def __init__(self, dset, root_drt=None, modal=None, dset_type='gallery'):
        if modal[:4] == 'peri':
            sz = (112, 112)
        elif modal[:4] == 'face':
            sz = (112, 112)
        
        self.ocular_root_dir = os.path.join(dset) #os.path.join(os.path.join(root_drt, dset, dset_type), modal[:4])
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


def prot_prot_verify(feature_extractor_p, generator_p, feature_extractor_u, generator_u, prot_path=None, face_path=None, emb_size = 512, root_drt=config.evaluation['verification'], device='cuda:1'):
    embedding_size = emb_size

    prot_dset = dataset(dset=prot_path, dset_type='gallery', root_drt = None, modal='periocular')
    face_dset = dataset(dset=face_path, dset_type='gallery', root_drt = None, modal='face')
    
    prot_dloader = torch.utils.data.DataLoader(prot_dset, batch_size=batch_size, num_workers=4)
    nof_prot_dset = len(prot_dset)
    nof_prot_iden = prot_dset.nof_identity
    prot_embedding_mat = torch.zeros((nof_prot_dset, embedding_size)).to(device)
    prot_label_mat = torch.zeros((nof_prot_dset, nof_prot_iden)).to(device)

    face_dloader = torch.utils.data.DataLoader(face_dset, batch_size=batch_size, num_workers=4)
    nof_face_dset = len(face_dset)
    nof_face_iden = face_dset.nof_identity
    face_embedding_mat = torch.zeros((nof_face_dset, embedding_size)).to(device)
    face_label_mat = torch.zeros((nof_face_dset, nof_face_iden)).to(device)

    feature_extractor_p = feature_extractor_p.eval().to(device)
    generator_p = generator_p.eval().to(device)
    feature_extractor_u = feature_extractor_u.eval().to(device)
    generator_u = generator_u.eval().to(device)

    with torch.no_grad():
        for i, (prot_ocular, prot_onehot, prot_lbl) in enumerate(prot_dloader):
            nof_prot_img = prot_ocular.shape[0]
            prot_ocular = prot_ocular.to(device)

            prot_feature = feature_extractor_p(prot_ocular)
            prot_feature = generator_p(prot_feature, prot_lbl)
            prot_onehot = prot_onehot.to(device)
            prot_lbl = prot_lbl.to(device)

            prot_embedding_mat[i*batch_size:i*batch_size+nof_prot_img, :] = prot_feature.detach().clone()                
            prot_label_mat[i*batch_size:i*batch_size+nof_prot_img, :] = prot_onehot


        for i, (face_ocular, face_onehot, face_lbl) in enumerate(face_dloader):
            nof_face_img = face_ocular.shape[0]
            face_ocular = face_ocular.to(device)

            face_feature = feature_extractor_u(face_ocular)
            face_feature = generator_u(face_feature, face_lbl)
            face_onehot = face_onehot.to(device)
            face_lbl = face_lbl.to(device)

            face_embedding_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_feature.detach().clone()
            face_label_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_onehot           

        ### roc
        face_embedding_mat /= torch.norm(face_embedding_mat, p=2, dim=1, keepdim=True)
        prot_embedding_mat /= torch.norm(prot_embedding_mat, p=2, dim=1, keepdim=True)

        score_mat = torch.matmul(face_embedding_mat, prot_embedding_mat.t()).cpu()
        gen_mat = torch.matmul(face_label_mat, prot_label_mat.t()).cpu()
        gen_r, gen_c = torch.where(gen_mat == 1)
        imp_r, imp_c = torch.where(gen_mat == 0)

        gen_score = score_mat[gen_r, gen_c].cpu().numpy()
        imp_score = score_mat[imp_r, imp_c].cpu().numpy()

        y_gen = np.ones(gen_score.shape[0])
        y_imp = np.zeros(imp_score.shape[0])

        score = np.concatenate((gen_score, imp_score))
        y = np.concatenate((y_gen, y_imp))

        # normalization scores into [ -1, 1]
        score_min = np.amin(score)
        score_max = np.amax(score)
        score = ( score - score_min ) / ( score_max - score_min )
        score = 2.0 * score - 1.0

        fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
        eer_dict['security'] = compute_eer(fpr_tmp, tpr_tmp)

    return eer_dict


def unprot_unprot_verify(feature_extractor_p, generator_p, feature_extractor_u, generator_u, unprot_path=None, face_path=None, emb_size = 512, root_drt=config.evaluation['verification'], device='cuda:1'):
    embedding_size = emb_size

    unprot_dset = dataset(dset=unprot_path, dset_type='gallery', root_drt = None, modal='periocular')
    face_dset = dataset(dset=face_path, dset_type='gallery', root_drt = None, modal='face')
    
    unprot_dloader = torch.utils.data.DataLoader(unprot_dset, batch_size=batch_size, num_workers=4)
    nof_unprot_dset = len(unprot_dset)
    nof_unprot_iden = unprot_dset.nof_identity
    unprot_embedding_mat = torch.zeros((nof_unprot_dset, embedding_size)).to(device)
    unprot_label_mat = torch.zeros((nof_unprot_dset, nof_unprot_iden)).to(device)

    face_dloader = torch.utils.data.DataLoader(face_dset, batch_size=batch_size, num_workers=4)
    nof_face_dset = len(face_dset)
    nof_face_iden = face_dset.nof_identity
    face_embedding_mat = torch.zeros((nof_face_dset, embedding_size)).to(device)
    face_label_mat = torch.zeros((nof_face_dset, nof_face_iden)).to(device)

    feature_extractor_p = feature_extractor_p.eval().to(device)
    generator_p = generator_p.eval().to(device)
    feature_extractor_u = feature_extractor_u.eval().to(device)
    generator_u = generator_u.eval().to(device)

    with torch.no_grad():
        for i, (unprot_ocular, unprot_onehot, unprot_lbl) in enumerate(unprot_dloader):
            nof_unprot_img = unprot_ocular.shape[0]
            unprot_ocular = unprot_ocular.to(device)

            unprot_feature = feature_extractor_u(unprot_ocular)
            unprot_feature = generator_u(unprot_feature, unprot_lbl)
            unprot_onehot = unprot_onehot.to(device)
            unprot_lbl = unprot_lbl.to(device)

            unprot_embedding_mat[i*batch_size:i*batch_size+nof_unprot_img, :] = unprot_feature.detach().clone()                
            unprot_label_mat[i*batch_size:i*batch_size+nof_unprot_img, :] = unprot_onehot


        for i, (face_ocular, face_onehot, face_lbl) in enumerate(face_dloader):
            nof_face_img = face_ocular.shape[0]
            face_ocular = face_ocular.to(device)

            face_feature = feature_extractor_u(face_ocular)
            face_feature = generator_u(face_feature, face_lbl)
            face_onehot = face_onehot.to(device)
            face_lbl = face_lbl.to(device)

            face_embedding_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_feature.detach().clone()
            face_label_mat[i*batch_size:i*batch_size+nof_face_img, :] = face_onehot           

        ### roc
        face_embedding_mat /= torch.norm(face_embedding_mat, p=2, dim=1, keepdim=True)
        unprot_embedding_mat /= torch.norm(unprot_embedding_mat, p=2, dim=1, keepdim=True)

        score_mat = torch.matmul(face_embedding_mat, unprot_embedding_mat.t()).cpu()
        gen_mat = torch.matmul(face_label_mat, unprot_label_mat.t()).cpu()
        gen_r, gen_c = torch.where(gen_mat == 1)
        imp_r, imp_c = torch.where(gen_mat == 0)

        gen_score = score_mat[gen_r, gen_c].cpu().numpy()
        imp_score = score_mat[imp_r, imp_c].cpu().numpy()

        y_gen = np.ones(gen_score.shape[0])
        y_imp = np.zeros(imp_score.shape[0])

        score = np.concatenate((gen_score, imp_score))
        y = np.concatenate((y_gen, y_imp))

        # normalization scores into [ -1, 1]
        score_min = np.amin(score)
        score_max = np.amax(score)
        score = ( score - score_min ) / ( score_max - score_min )
        score = 2.0 * score - 1.0

        fpr_tmp, tpr_tmp, _ = roc_curve(y, score)
        eer_dict['security'] = compute_eer(fpr_tmp, tpr_tmp)

    return eer_dict


if __name__ == '__main__':
    embd_dim = 512
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    load_model_path_fe_p = './models/best_feature_extractor/FSB-HashNet.pth'
    feature_extractor_p = net1.FSB_Hash_Net(embedding_size = 1024, do_prob = 0.0).eval().to(device)
    feature_extractor_p = load_model.load_pretrained_network(feature_extractor_p, load_model_path_fe_p, device = device)    
    load_model_path_p = load_model_path_fe_p.replace('best_feature_extractor', 'best_generator')
    generator_p = net1.Hash_Generator(embedding_size = 1024, device=device, out_embedding_size=512).eval().to(device)
    generator_p = load_model.load_pretrained_network(generator_p, load_model_path_p, device = device)

    load_model_path_fe_u = './models/best_feature_extractor/FSB-HashNet_Baseline.pth'
    feature_extractor_u = net2.FSB_Hash_Net(embedding_size = 1024, do_prob = 0.0).eval().to(device)
    feature_extractor_u = load_model.load_pretrained_network(feature_extractor_u, load_model_path_fe_u, device = device)
    load_model_path_u = load_model_path_fe_u.replace('best_feature_extractor', 'best_generator')
    generator_u = net2.Hash_Generator(embedding_size = 1024, device=device, out_embedding_size=512).eval().to(device)
    generator_u = load_model.load_pretrained_network(generator_u, load_model_path_u, device = device)

    unprot_face_path = '/home/tiongsik/Python/conditional_biometrics/data/synthetic_images/IDiff-Face_samples/FSB-HashNet_generated/Main_Unprotected/'
    prot_face_path = '/home/tiongsik/Python/conditional_biometrics/data/synthetic_images/IDiff-Face_samples/FSB-HashNet_generated/Main_Protected/'
    real_face_path = '/home/tiongsik/Python/conditional_biometrics/data/visualization/security_db/real/face/'

    real_vs_prot = prot_prot_verify(feature_extractor_p, generator_p, feature_extractor_p, generator_p, prot_path = prot_face_path, face_path = real_face_path, emb_size = embd_dim, root_drt = config.evaluation['verification'], device = device)
    real_vs_prot = copy.deepcopy(real_vs_prot)          
    print('Bona Fide (Protected) vs Protected SAR:', real_vs_prot)    
    
    real_vs_unprot = unprot_unprot_verify(feature_extractor_u, generator_u, feature_extractor_u, generator_u, unprot_path = unprot_face_path, face_path = real_face_path, emb_size = embd_dim, root_drt = config.evaluation['verification'], device = device)
    real_vs_unprot = copy.deepcopy(real_vs_unprot)          
    print('Bona Fide (Unprotected) vs Unprotected SAR:', real_vs_unprot)