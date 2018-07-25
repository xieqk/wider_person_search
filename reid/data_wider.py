from __future__ import print_function, absolute_import
import os
import os.path as osp
import glob

class wider_train(object):
    dataset_dir = 'wider_train'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> wider-reid loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1][:4])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1][:4])
            camid = int(img_path.split('/')[-1][5])
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        
        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


class wider_exfeat(object):
    dataset_dir = 'wider_exfeat'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        val, num_val_imgs = self._process_dir(self.val_dir, relabel=False)
        test, num_test_imgs = self._process_dir(self.test_dir, relabel=False)
        num_total_imgs = num_val_imgs + num_test_imgs

        print("=> wider_exfeat loaded")
        print("Dataset statistics:")
        print("  ------------------------")
        print("  subset     | # images")
        print("  ------------------------")
        print("  validation | {:8d}".format(num_val_imgs))
        print("  test       | {:8d}".format(num_test_imgs))
        print("  ------------------------")
        print("  total      | {:8d}".format(num_total_imgs))
        print("  ------------------------")

        self.val = val
        self.test = test

        self.num_val_imgs = num_val_imgs
        self.num_test_imgs = num_test_imgs

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
    
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        dataset = []
        for img_path in img_paths:
            pid = img_path.split('/')[-1][:-4]
            dataset.append((img_path, pid, 0))
        
        num_imgs = len(dataset)
        return dataset, num_imgs

"""Create dataset"""

__factory = {
    'wider_train': wider_train,
    'wider_exfeat': wider_exfeat,
}

def get_names():
    return list(__factory.keys())

def init_img_dataset(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))
    return __factory[name](**kwargs)
