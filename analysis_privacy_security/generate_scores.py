import torch
import yaml
import numpy as np
import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath('.'))
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from data import data_loader
from configs import datasets_config as config
import network.fsb_hash_net as net
from network import load_model
from scipy.spatial import distance
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')

def run():
    torch.multiprocessing.freeze_support()
    embd_dim = 512
    method = 'FSB_HashNet'
    folder = os.path.join('./analysis_privacy_security', method)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Change dataset at this portion of code
    face_gallery_dir = config.ethnic['face_gallery']
    face_val_loader, face_val_set = data_loader.gen_data(face_gallery_dir, 'test', type='face')

    load_model_path_fe = './models/best_feature_extractor/FSB-HashNet.pth'
    feature_extractor = net.FSB_Hash_Net(embedding_size=1024, do_prob=0.0).eval().to(device)
    feature_extractor = load_model.load_pretrained_network(feature_extractor, load_model_path_fe, device = device)
    
    load_model_path = load_model_path_fe.replace('best_feature_extractor', 'best_generator')
    generator = net.Hash_Generator(embedding_size=1024, device=device, out_embedding_size=embd_dim).eval().to(device)
    generator = load_model.load_pretrained_network(generator, load_model_path, device = device)

    feature_extractor = feature_extractor.eval().to(device)
    generator = generator.eval().to(device)

    all_vectors_g1 = torch.tensor([])
    all_vectors_g2 = torch.tensor([])
    all_vectors_g3 = torch.tensor([])
    all_vectors_g4 = torch.tensor([])
    all_vectors_g5 = torch.tensor([])
    all_labels = torch.tensor([], dtype = torch.int64)

    # get features and mated samples
    for x, y in face_val_loader:  
          
        x = x.to(device)
        y = y.to(device)

        feature = feature_extractor(x)
        hashed = generator(feature, y, device).to(device)
        hashed_2 = generator(feature, y+7, device).to(device)
        hashed_3 = generator(feature, y+14, device).to(device)
        hashed_4 = generator(feature, y+21, device).to(device)
        hashed_5 = generator(feature, y+28, device).to(device)
        
        all_vectors_g1 = torch.cat((all_vectors_g1, hashed.detach().cpu()), dim=0)
        all_vectors_g2 = torch.cat((all_vectors_g2, hashed_2.detach().cpu()), dim=0)
        all_vectors_g3 = torch.cat((all_vectors_g3, hashed_3.detach().cpu()), dim=0)
        all_vectors_g4 = torch.cat((all_vectors_g4, hashed_4.detach().cpu()), dim=0)
        all_vectors_g5 = torch.cat((all_vectors_g5, hashed_5.detach().cpu()), dim=0)

        all_labels = torch.cat((all_labels, y.detach().cpu()))

        del x, y

    # Genuine
    genuine_scores = torch.tensor([])

    for i in tqdm(range(0, len(all_labels))):
        for j in range(i, len(all_labels)):
            if i<j and all_labels[i]==all_labels[j]:
                gen_score1 = distance.cosine(all_vectors_g1[i], all_vectors_g1[j])
                gen_score2 = distance.cosine(all_vectors_g2[i], all_vectors_g2[j])
                gen_score3 = distance.cosine(all_vectors_g3[i], all_vectors_g3[j])
                gen_score4 = distance.cosine(all_vectors_g4[i], all_vectors_g4[j])
                gen_score5 = distance.cosine(all_vectors_g5[i], all_vectors_g5[j])
                genuine_scores = torch.cat((genuine_scores, torch.tensor(gen_score1).unsqueeze(0)))
                genuine_scores = torch.cat((genuine_scores, torch.tensor(gen_score2).unsqueeze(0)))
                genuine_scores = torch.cat((genuine_scores, torch.tensor(gen_score3).unsqueeze(0)))
                genuine_scores = torch.cat((genuine_scores, torch.tensor(gen_score4).unsqueeze(0)))
                genuine_scores = torch.cat((genuine_scores, torch.tensor(gen_score5).unsqueeze(0)))

    np.savetxt('./analysis_privacy_security/' + str(method) + '/genuine.txt', genuine_scores)

    # # Imposter
    imposter_scores = torch.tensor([])

    # unique_labels = torch.unique(all_labels)
    for k in tqdm(range(0, len(all_labels))):
        for m in range(k, len(all_labels)):
            if all_labels[k]<all_labels[m]:
                imp_score = distance.cosine(all_vectors_g1[k], all_vectors_g1[m])
                imposter_scores = torch.cat((imposter_scores, torch.tensor(imp_score).unsqueeze(0)))
    
    np.savetxt('./analysis_privacy_security/' + str(method) + '/imposter.txt', imposter_scores)    

    print('Genuine & Imposter Done!')


    # Mated score
    mated_scores = torch.tensor([])
    non_mated_scores = torch.tensor([])
    
    for i in tqdm(range(0, len(all_labels))):
        for j in range(i, len(all_labels)):
            if i<j and all_labels[i]==all_labels[j]:
                m_score1 = distance.cosine(all_vectors_g1[i], all_vectors_g2[j])
                m_score2 = distance.cosine(all_vectors_g1[i], all_vectors_g3[j])
                m_score3 = distance.cosine(all_vectors_g1[i], all_vectors_g4[j])
                m_score4 = distance.cosine(all_vectors_g1[i], all_vectors_g5[j])
                m_score5 = distance.cosine(all_vectors_g2[i], all_vectors_g3[j])
                m_score6 = distance.cosine(all_vectors_g2[i], all_vectors_g4[j])
                m_score7 = distance.cosine(all_vectors_g2[i], all_vectors_g5[j])
                m_score8 = distance.cosine(all_vectors_g3[i], all_vectors_g4[j])
                m_score9 = distance.cosine(all_vectors_g3[i], all_vectors_g5[j])
                m_score10 = distance.cosine(all_vectors_g4[i], all_vectors_g5[j])                
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score1).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score2).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score3).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score4).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score5).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score6).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score7).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score8).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score9).unsqueeze(0)))
                mated_scores = torch.cat((mated_scores, torch.tensor(m_score10).unsqueeze(0)))

    np.savetxt('./analysis_privacy_security/' + str(method) + '/mated.txt', mated_scores)

    # Non-Mated score
    for k in tqdm(range(0, len(all_labels))):
        for m in range(k, len(all_labels)):
            if all_labels[k]<all_labels[m]:

                nm_score1 = distance.cosine(all_vectors_g1[k], all_vectors_g2[m])
                non_mated_scores = torch.cat((non_mated_scores, torch.tensor(nm_score1).unsqueeze(0)))

    np.savetxt('./analysis_privacy_security/' + str(method) + '/nonmated.txt', non_mated_scores)

    print('Mated & Nonmated Done!')

if __name__=='__main__':
    run()