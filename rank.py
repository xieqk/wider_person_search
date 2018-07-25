import json
import numpy as np
import argparse
import os.path as osp

from eval import eval
from utils.pkl import my_unpickle
from scipy.spatial.distance import pdist, squareform

def load_json(name):
    with open(name) as f:
        data = json.load(f)
        return data

def load_face(face_data):
    face_dict = {}
    movie_list = []
    for movie, info in face_data.items():
        movie_list.append(movie)
        casts = info['cast']
        candidates = info['candidates']
        
        cast_ids, cast_ffeats = [], []
        for cast in casts:
            cast_ffeats.append(cast['ffeat'])
            cast_ids.append(cast['id'])
        cast_ffeats = np.array(cast_ffeats)

        candi_f_ids, candi_f_ffeats = [], []
        for candidate in candidates:
            if candidate['fbbox'] is not None:
                candi_f_ids.append(candidate['id'])
                candi_f_ffeats.append(candidate['ffeat'])
        candi_f_ffeats = np.array(candi_f_ffeats)

        face_dict.update(
            {
                movie:{
                    'cast_ids':cast_ids, 
                    'cast_ffeats':cast_ffeats, 
                    'candi_f_ids': candi_f_ids, 
                    'candi_f_ffeats':candi_f_ffeats, 
                }
            }
        )
    return face_dict, movie_list

def load_reid(reid_data):
    reid_dict_tmp = {}
    reid_dict = {}
    for key, value in reid_data.items():
        movie = key[:9]
        if movie not in reid_dict_tmp.keys():
            reid_dict_tmp.update({movie:{}})
        reid_dict_tmp[movie].update({key:value})
    for movie, info in reid_dict_tmp.items():
        candi_ids, candi_feats = [], []
        for candi_id, candi_feat in info.items():
            candi_ids.append(candi_id)
            candi_feats.append(candi_feat)
        candi_feats = np.array(candi_feats)

        reid_dict.update(
            {
                movie:{
                    'candi_ids':candi_ids,
                    'candi_feats':candi_feats
                }
            }
        )
    return reid_dict

def load_reid_4(reid_data1, reid_data2, reid_data3, reid_data4):
    reid_dict_tmp = {}
    reid_dict = {}
    for key, value in reid_data1.items():
        movie = key[:9]
        if movie not in reid_dict_tmp.keys():
            reid_dict_tmp.update({movie:{}})
        feat1 = value
        feat2 = reid_data2[key]
        feat3 = reid_data3[key]
        feat4 = reid_data4[key]
        feat1 = np.array(feat1)
        feat2 = np.array(feat2)
        feat3 = np.array(feat3)
        feat4 = np.array(feat4)
        feat = np.hstack((feat1, feat2, feat3, feat4))
        reid_dict_tmp[movie].update({key:feat})
    for movie, info in reid_dict_tmp.items():
        candi_ids, candi_feats = [], []
        for candi_id, candi_feat in info.items():
            candi_ids.append(candi_id)
            candi_feats.append(candi_feat)
        candi_feats = np.array(candi_feats)
        
        reid_dict.update(
            {
                movie:{
                    'candi_ids':candi_ids,
                    'candi_feats':candi_feats
                }
            }
        )
    return reid_dict

def multi_face_recall(cast_candi_filter, candi_f_ids, candi_candi_fsim):
    rows, cols = cast_candi_filter.shape

    result = np.zeros((rows, cols))
    for i in range(rows):
        if cast_candi_filter[i].sum() == 0:
            continue
        for j in range(cols):
            sims = []
            for idx, flag in enumerate(cast_candi_filter[i]):
                if flag != 0:
                    sims.append(candi_candi_fsim[j, idx])
            sims = np.array(sims)
            max_sim = sims.max()
            if max_sim > 0.5:
                result[i,j] = 1
    recall_num = (result-cast_candi_filter).sum()
    return result, recall_num

def multi_search(cast_candi_filter, candi_f_ids, candi_ids, candi_candi_dist):
    rows, cols = cast_candi_filter.shape
    new_rows, new_cols = rows, len(candi_ids)
    assert cols <= new_cols

    pre_query_inds = []
    for i in range(rows):
        pre_query_inds.append([])
        for j in range(cols):
            if cast_candi_filter[i,j] != 0:
                idx = candi_ids.index(candi_f_ids[j])
                pre_query_inds[i].append(idx)
    
    result = np.full((new_rows, new_cols), 9999)
    for i in range(new_rows):
        if len(pre_query_inds[i]) == 0:
            continue
        for j in range(new_cols):
            dists = []
            for idx in pre_query_inds[i]:
                dists.append(candi_candi_dist[idx, j])
            dists = np.array(dists)
            min_dist = dists.min()
            result[i,j] = min_dist
    
    return result

def rank(movie_face, movie_reid):
    cast_ids, cast_ffeats = movie_face['cast_ids'], movie_face['cast_ffeats']
    candi_f_ids, candi_f_ffeats = movie_face['candi_f_ids'], movie_face['candi_f_ffeats']
    candi_ids, candi_feats = movie_reid['candi_ids'], movie_reid['candi_feats']
    movie_rank = {cast_id:[] for cast_id in cast_ids}

    cast_candi_fsim = np.dot(cast_ffeats, candi_f_ffeats.T)
    candi_candi_fsim = np.dot(candi_f_ffeats, candi_f_ffeats.T)
    # print(candi_feats.shape)
    candi_candi_dist = pdist(candi_feats, 'euclidean')
    candi_candi_dist = squareform(candi_candi_dist)
    assert cast_candi_fsim.shape[0] == len(cast_ids) and cast_candi_fsim.shape[1] == len(candi_f_ids)

    # get cast_candi_flag
    row_num, col_num = cast_candi_fsim.shape
    cast_candi_filter = np.zeros((row_num, col_num))
    for i, candi_id in enumerate(candi_f_ids):
        sim = cast_candi_fsim.T[i].copy()
        max_ind = np.argsort(sim)[-1]
        if sim[max_ind] > 0.29:
            cast_candi_filter[max_ind, i] = 1
            movie_rank[cast_ids[max_ind]].append(candi_id)
    
    cast_candi_filter, recall_num = multi_face_recall(cast_candi_filter, candi_f_ids, candi_candi_fsim)

    # multi query search
    tmp_dist = multi_search(cast_candi_filter, candi_f_ids, candi_ids, candi_candi_dist) 
    
    # add res
    for i, cast_id in enumerate(cast_ids):
        inds = np.argsort(tmp_dist[i])
        for idx in inds:
            if candi_ids[idx] not in movie_rank[cast_id]:
                movie_rank[cast_id].append(candi_ids[idx])

    return movie_rank, recall_num

def rank2txt(rank, file_name):
    with open(file_name, 'w') as f:
        for cast_id, candi_ids in rank.items():
            line = '%s %s\n'%(cast_id, ','.join(candi_ids))
            f.write(line)

def rank_eval(res, label):
    all_ap = eval(res, label)
    return np.array(all_ap)

def main(args):
    if args.is_test =='0':
        face_feat_name = 'face_em_val.pkl'
        reid_feat_name_resnet101 = 'reid_em_val_resnet101.pkl'
        reid_feat_name_densenet121 = 'reid_em_val_densenet121.pkl'
        reid_feat_name_seresnet101 = 'reid_em_val_seresnet101.pkl'
        reid_feat_name_seresnext101 = 'reid_em_val_seresnext101.pkl'
    else:
        face_feat_name = 'face_em_test.pkl'
        reid_feat_name_resnet101 = 'reid_em_test_resnet101.pkl'
        reid_feat_name_densenet121 = 'reid_em_test_densenet121.pkl'
        reid_feat_name_seresnet101 = 'reid_em_test_seresnet101.pkl'
        reid_feat_name_seresnext101 = 'reid_em_test_seresnext101.pkl'
    
    print('Load features from pkl ...')
    face_pkl = my_unpickle(osp.join('./features', face_feat_name))
    face_dict, movie_list = load_face(face_pkl)
    if args.arch is None:
        reid_pkl_resnet101 = my_unpickle(osp.join('./features', reid_feat_name_resnet101))
        reid_pkl_densenet121 = my_unpickle(osp.join('./features', reid_feat_name_densenet121))
        reid_pkl_seresnet101 = my_unpickle(osp.join('./features', reid_feat_name_seresnet101))
        reid_pkl_seresnext101 = my_unpickle(osp.join('./features', reid_feat_name_seresnext101))
        reid_dict = load_reid_4(reid_pkl_resnet101, reid_pkl_densenet121, reid_pkl_seresnet101, reid_pkl_seresnext101)
    elif args.arch == 'resnet101':
        reid_pkl = my_unpickle(osp.join('./features', reid_feat_name_resnet101))
        reid_dict = load_reid(reid_pkl)
    elif args.arch == 'densenet121':
        reid_pkl = my_unpickle(osp.join('./features', reid_feat_name_densenet121))
        reid_dict = load_reid(reid_pkl)
    elif args.arch == 'seresnet101':
        reid_pkl = my_unpickle(osp.join('./features', reid_feat_name_seresnet101))
        reid_dict = load_reid(reid_pkl)
    elif args.arch == 'seresnext101':
        reid_pkl = my_unpickle(osp.join('./features', reid_feat_name_seresnext101))
        reid_dict = load_reid(reid_pkl)
    print('Done !')

    rank_list = {}
    movie_num = len(movie_list)
    for i, movie in enumerate(movie_list):
        movie_face = face_dict[movie]
        movie_reid = reid_dict[movie]
        movie_rank, recall_num = rank(movie_face, movie_reid)
        print('movie: %s, %d/%d, recall num: %d'%(movie, i+1, movie_num, recall_num))
        rank_list.update(movie_rank)

    if args.is_test == '1':
        rank2txt(rank_list, 'test_rank.txt')
    else:
        rank2txt(rank_list, 'val_rank.txt')
        all_ap = rank_eval('val_rank.txt', 'val_label.json')
        print(np.sort(all_ap)[::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-test', type=str, default='0', choices=['0', '1'])
    parser.add_argument('-a', '--arch', type=str, default=None, choices=['resnet101', 'densenet121', 'seresnet101', 'seresnext101'])
    args = parser.parse_args()
    main(args)


