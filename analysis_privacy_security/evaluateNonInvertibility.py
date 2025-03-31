import os, sys
sys.path.insert(0, os.path.abspath('.'))

import torch
import os, glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import pairwise
from data import data_loader
from network import load_model
import network.fsb_hash_net as net1
import network.fsb_hash_net_baseline as net2
import seaborn as sns
import pylab
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

# plot histogram for intra/inter modal intra/inter class comparison
def feature_extraction(feature_extractor, generator, data_loader, device='cuda:0', protect_flag=True):    
    emb = torch.tensor([])
    lbl = torch.tensor([], dtype = torch.int64)

    feature_extractor = feature_extractor.eval().to(device)
    generator = generator.eval().to(device)
    with torch.no_grad():        
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            x = feature_extractor(x)
            if protect_flag == True:
                x = generator(x, y)
            
            emb = torch.cat((emb, x.detach().cpu(),), 0)
            lbl = torch.cat((lbl, y))            
            
            del x, y
            time.sleep(0.0001)

    # print('Set Capacity\t: ', emb.size())
    assert(emb.size()[0] == lbl.size()[0])
    
    del data_loader
    time.sleep(0.0001)

    del feature_extractor, generator
    
    return emb, lbl


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return pairwise.cosine_similarity(a, b)


def plot_hist_unprotected_protected(protected, unprotected, figureFile):
    unprotected, protected = np.array(unprotected).ravel(), np.array(protected).ravel()
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # plt.rcParams.update({'font.size': 17, 'legend.fontsize': 15})  
    
    plt.figure()
    plt.xticks(np.arange(-1, 1, 0.1))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sns.kdeplot(protected, fill=True, alpha=0.8, label='Protected vs. Bona Fide', linewidth=2, color=sns.xkcd_rgb["pale red"], ax=ax1)
    sns.kdeplot(unprotected, fill=True, alpha=0.8, label='Unprotected vs. Bona Fide', linewidth=2, color=sns.xkcd_rgb["medium green"], ax=ax2)    
    ax1.tick_params(axis ='y')

    ax2.axes.get_yaxis().set_visible(False)
    fig.legend(loc="upper left", bbox_to_anchor=(0.13, 0), prop={'size': 10}, ncol=2)
    sns.set_context("paper",font_scale=1.7, rc={"lines.linewidth": 2.5})
    sns.set_style("white")    
    plt.title('Non-Invertibility Analysis')
    ax1.set_ylabel("Probability Density")
    ax1.set_xlabel("Cosine Similarity Score")
    pylab.savefig(figureFile, bbox_inches='tight')


# Accepts (un)protected features and face features, calculating similarity of similar subjects regardless of instance
def inter_model(unpr_features, unpr_label, face_features, face_label, cls):
    dist = torch.tensor([])

    for i in torch.unique(unpr_label):
        unpr_indices = np.array(np.where(unpr_label == i)).ravel()
        non_unpr_indices = np.array(np.where(face_label != i)).ravel()

        sims = torch.Tensor(cosine_sim(unpr_features[unpr_indices], face_features[unpr_indices]).ravel())
        dist = torch.cat((dist, sims), 0)

    return sims.reshape(-1)


def extract(feature_extractor_model, generator, data_path, device, modal, protect_flag):
    data_load, data_set = data_loader.gen_data(data_path, 'test', type=modal, aug='False')
    feat, labl = feature_extraction(feature_extractor_model, generator, data_load, device = device, protect_flag = protect_flag) 

    return feat, labl, data_load, data_set


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    protect_flag = True
    method = 'fsb_hashnet'

    # get models: FSB-HashNet - protected, FSB-HashNet (Baseline) - unprotected
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

    protected = './data/non_invertibility/recon_protected/'
    unprotected = './data/non_invertibility/recon_unprotected/'
    face = './data/non_invertibility/bona_fide/'
    
    prot_fea, prot_label, prot_data_load, prot_data_set = extract(feature_extractor_p, generator_p, protected, device, 'face', True)
    unpro_fea, unpro_label, unpro_data_load, unpro_data_set = extract(feature_extractor_u, generator_u, unprotected, device, 'face', False)
    facep_fea, facep_label, facep_data_load, facep_data_set = extract(feature_extractor_p, generator_p, face, device, 'face', True)
    faceu_fea, faceu_label, faceu_data_load, faceu_data_set = extract(feature_extractor_u, generator_u, face, device, 'face', False)

    unprotected_vs_bonafide = inter_model(unpro_fea, unpro_label, faceu_fea, faceu_label, cls='intra')
    protected_vs_bonafide = inter_model(prot_fea, prot_label, facep_fea, facep_label, cls='intra')
    figureFile = './graphs/analysis_privacy_security/' + str(method) + '/image_noninvertibility.pdf'

    plot_hist_unprotected_protected(protected_vs_bonafide, unprotected_vs_bonafide, figureFile)