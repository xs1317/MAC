from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False
            # inductive=True
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        # new addition
        # if phase == 'train':
        #     self.inductive = inductive
        # else:
        #     self.inductive = False

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data


class MultiAttrCompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False
            # inductive=True
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        # new addition
        # if phase == 'train':
        #     self.inductive = inductive
        # else:
        #     self.inductive = False

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.trainpair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}

        self.ao_pair_num = [0] * len(self.train_pairs)
        self.a_num = [0] * len(self.attrs)

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data


        print('# train pairs: %d | # val pairs: %d | # test pairs: %d | # attrs: %d | # objs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs),len(self.attrs),len(self.objs)  ))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))

        train_data, val_data, test_data = [], [], []

        
        for instance in data:
            image, attrs, obj, settype = instance['image'], instance[
                'attrs'], instance['obj'], instance['set']

            if len(attrs) == 0  or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
            
            attrs_list = [0] * len(self.attrs)
            
            for a in attrs:
                attrs_list[self.attr2idx[a]] = 1
                
            data_i = [image, attrs, obj,attrs_list] #,pairs_list]
            if settype == 'train':
                pairs_list = [0] * len(self.train_pairs)
                for a in attrs:
                    a_idx = self.attr2idx[a]
                    ao_pair_idx = self.trainpair2idx[(a,obj)]
                    self.ao_pair_num[ao_pair_idx] += 1
                    self.a_num[a_idx] += 1
                    pairs_list[ao_pair_idx] = 1
                
                data_i.append(pairs_list)

                train_data.append(data_i)
            elif settype == 'val':
                pairs_list = [0] * len(self.pairs)
                for a in attrs:
                    pairs_list[self.pair2idx[(a,obj)]] = 1
                
                data_i.append(pairs_list)
                val_data.append(data_i)
            else:
                pairs_list = [0] * len(self.pairs)
                for a in attrs:
                    pairs_list[self.pair2idx[(a,obj)]] = 1
                
                data_i.append(pairs_list)
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                # convert a1 a2 a3 o1 -> a1 o1 ; a2 o1 ; a3 o1
                pairs = [[attr,t[-1]]  for t in pairs for attr in t[0:-1]]
                pairs = list(set(map(tuple, pairs)))

            objs = list(set([t[-1] for t in pairs]))
            attrs = list(set([t[0:-1] for t in pairs]))
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs = sorted(list(set([item for sublist in tr_attrs for item in sublist] + 
                                    [item for sublist in vl_attrs for item in sublist] +
                                    [item for sublist in ts_attrs for item in sublist])))
         
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))

        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attrs, obj,attr_list,pair_list = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, 
                torch.Tensor(attr_list), 
                self.obj2idx[obj], 
                torch.Tensor(pair_list)
            ]
        else:
            data = [
                img, 
                torch.Tensor(attr_list), 
                self.obj2idx[obj], 
                torch.Tensor(pair_list)
            ]

        return data

    def __len__(self):
        return len(self.data)


class MultiAttrCompositionDatasetWhite(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            part = "all"     # white/no_white/all
            # inductive=True
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.part = part


        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.trainpair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}

        self.ao_pair_num = [0] * len(self.train_pairs)
        self.a_num = [0] * len(self.attrs)

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data


        print('# train pairs: %d | # val pairs: %d | # test pairs: %d | # attrs: %d | # objs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs),len(self.attrs),len(self.objs)  ))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))

        white_back_ground_index = torch.load(self.root+'/{}_white_background_index.t7'.format(self.split))
        all_index = list(range(len(data)))

        train_data, val_data, test_data = [], [], []

        if self.part == "white":
            part_index = white_back_ground_index
        elif self.part == "no_white":
            part_index = list(set(all_index)-set(white_back_ground_index))
        elif self.part == "all":
            part_index = all_index

        for idx in part_index:
            instance = data[idx]
            image, attrs, obj, settype = instance['image'], instance[
                'attrs'], instance['obj'], instance['set']

            if len(attrs) == 0  or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
            
            attrs_list = [0] * len(self.attrs)
            
            for a in attrs:
                attrs_list[self.attr2idx[a]] = 1
                
            data_i = [image, attrs, obj,attrs_list] #,pairs_list]
            if settype == 'train':
                pairs_list = [0] * len(self.train_pairs)
                for a in attrs:
                    a_idx = self.attr2idx[a]
                    ao_pair_idx = self.trainpair2idx[(a,obj)]
                    self.ao_pair_num[ao_pair_idx] += 1
                    self.a_num[a_idx] += 1
                    pairs_list[ao_pair_idx] = 1
                
                data_i.append(pairs_list)

                train_data.append(data_i)
            elif settype == 'val':
                pairs_list = [0] * len(self.pairs)
                for a in attrs:
                    pairs_list[self.pair2idx[(a,obj)]] = 1
                
                data_i.append(pairs_list)
                val_data.append(data_i)
            else:
                pairs_list = [0] * len(self.pairs)
                for a in attrs:
                    pairs_list[self.pair2idx[(a,obj)]] = 1
                
                data_i.append(pairs_list)
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                # convert a1 a2 a3 o1 -> a1 o1 ; a2 o1 ; a3 o1
                pairs = [[attr,t[-1]]  for t in pairs for attr in t[0:-1]]
                pairs = list(set(map(tuple, pairs)))

            objs = list(set([t[-1] for t in pairs]))
            attrs = list(set([t[0:-1] for t in pairs]))
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs = sorted(list(set([item for sublist in tr_attrs for item in sublist] + 
                                    [item for sublist in vl_attrs for item in sublist] +
                                    [item for sublist in ts_attrs for item in sublist])))
         
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))

        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attrs, obj,attr_list,pair_list = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, 
                torch.Tensor(attr_list), 
                self.obj2idx[obj], 
                torch.Tensor(pair_list)
            ]
        else:
            data = [
                img, 
                torch.Tensor(attr_list), 
                self.obj2idx[obj], 
                torch.Tensor(pair_list)
            ]

        return data

    def __len__(self):
        return len(self.data)


class CrossDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            mac_attrs=None,
            mac_objs=None,
            seen_pairs = None
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        if mac_attrs!=None and mac_objs!=None:
            self.attrs = mac_attrs
            self.objs = mac_objs
        else:
            self.attrs = None
            self.objs = None

        self.pairs, self.train_pairs, self.val_pairs, self.test_pairs,self.intersect_attrs,self.intersect_objs = self.parse_split()

        if self.objs == None or self.attrs == None:
            self.objs = self.intersect_objs
            self.attrs = self.intersect_attrs

        self.ao_pair_num = [0] * len(self.train_pairs)
        self.a_num = [0] * len(self.attrs)
        
        if seen_pairs != None:
            self.train_pairs = seen_pairs

        if self.open_world:
            self.pairs = list(product(self.intersect_attrs, self.intersect_objs))

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.trainpair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}

        self.train_data, self.val_data, self.test_data = self.get_split_info()



        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )


    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))

        train_data, val_data, test_data = [], [], []

        
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
            
            attrs_list = [0] * len(self.attrs)
            
            attrs_list[self.attr2idx[attr]] = 1
                
            data_i = [image,  attr, obj,attrs_list]

            attrs = [attr]
            if settype == 'train':
                pairs_list = [0] * len(self.train_pairs)
                for a in attrs:
                    a_idx = self.attr2idx[a]
                    ao_pair_idx = self.trainpair2idx[(a,obj)]
                    pairs_list[ao_pair_idx] = 1
                    self.ao_pair_num[ao_pair_idx] += 1
                    self.a_num[a_idx] += 1
                data_i.append(pairs_list)

                train_data.append(data_i)
            elif settype == 'val':
                pairs_list = [0] * len(self.pairs)
                for a in attrs:
                    pairs_list[self.pair2idx[(a,obj)]] = 1
                
                data_i.append(pairs_list)
                val_data.append(data_i)
            else:
                pairs_list = [0] * len(self.pairs)
                for a in attrs:
                    pairs_list[self.pair2idx[(a,obj)]] = 1
                
                data_i.append(pairs_list)
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        
        # preserve the pairs with the attr/obj in mac
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            

            
            if self.attrs != None and self.objs != None:
                pairs = [i for i in pairs if i[0] in self.attrs and i[1] in self.objs]

            attrs,objs = zip(*pairs)
            return  pairs,attrs,objs

        tr_pairs,tr_attrs,tr_objs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_pairs,vl_attrs,vl_objs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_pairs,ts_attrs,ts_objs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        intersect_attrs,intersect_objs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(list(set(tr_objs + vl_objs + ts_objs)))
        

        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_pairs, tr_pairs, vl_pairs, ts_pairs,intersect_attrs,intersect_objs

    def __getitem__(self, index):
        image, attrs, obj,attr_list,pair_list = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, 
                torch.Tensor(attr_list), 
                self.obj2idx[obj], 
                torch.Tensor(pair_list)
            ]
        else:
            data = [
                img, 
                torch.Tensor(attr_list), 
                self.obj2idx[obj], 
                torch.Tensor(pair_list)
            ]

        return data



    def __len__(self):
        return len(self.data)
    
