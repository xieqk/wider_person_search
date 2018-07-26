import json
import os
import os.path as osp
import cv2

trainval_root = '/data2/xieqk/wider/person_search_trainval'
test_root = '/data2/xieqk/wider/person_search_test'

def load_json(name):
    with open(name) as f:
        data = json.load(f)
        return data

def check_path(path):
    if not osp.exists(path):
        os.makedirs(path)
        print('path not exist, mkdir:', path)

if __name__ == '__main__':
    val = load_json(osp.join(trainval_root, 'val.json'))
    test = load_json(osp.join(test_root, 'test.json'))

    val_num, test_num = len(val.keys()), len(test.keys())
    val_cnt, test_cnt = 0, 0
    chk_val, chk_test = False, False
    for movie, info in val.items():
        val_cnt += 1
        candidates = info['candidates']
        candi_len = len(candidates)
        for i, candidate in enumerate(candidates):
            print('val: %d/%d, test: %d/%d ... %s %d/%d'%(val_cnt, val_num, test_cnt, test_num, candidate['img'], i+1, candi_len))
            pid = candidate['id']
            save_name = '%s.jpg'%pid
            img_path = osp.join(trainval_root, 'val', candidate['img'])
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            bbox = candidate['bbox']
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[0] + bbox[2] > w:
                bbox[2] = w - bbox[0]
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[1] + bbox[3] > h:
                bbox[3] = h - bbox[1]
            crop = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            crop = cv2.resize(crop, (128, 256))
            if chk_val == False:
                check_path(osp.join('./', 'data', 'wider_exfeat', 'val'))
                chk_val = True
            cv2.imwrite(osp.join('./data/wider_exfeat/val', save_name), crop)
    for movie, info in test.items():
        test_cnt += 1
        candidates = info['candidates']
        candi_len = len(candidates)
        for i, candidate in enumerate(candidates):
            print('val: %d/%d, test: %d/%d ... %s %d/%d'%(val_cnt, val_num, test_cnt, test_num, candidate['img'], i+1, candi_len))
            pid = candidate['id']
            save_name = '%s.jpg'%pid
            img_path = osp.join(test_root, 'test', candidate['img'])
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            bbox = candidate['bbox']
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[0] + bbox[2] > w:
                bbox[2] = w - bbox[0]
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[1] + bbox[3] > h:
                bbox[3] = h - bbox[1]
            crop = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            crop = cv2.resize(crop, (128, 256))
            if chk_test == False:
                check_path(osp.join('./', 'data', 'wider_exfeat', 'test'))
                chk_test = True
            cv2.imwrite(osp.join('./data/wider_exfeat/test', save_name), crop)