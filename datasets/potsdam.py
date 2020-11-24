"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


Mapillary Dataset Loader
"""
import os
import json

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder
from datasets import uniform
from PIL import Image
import numpy as np

def make_test_dataset(root, quality, mode):
    """
    Create File List
    """
    assert (quality == 'semantic' and mode in ['train', 'val'])
    img_dir_name = None
    if quality == 'semantic':
        if mode == 'train':
            img_dir_name = 'train_all'
        if mode == 'val':
            img_dir_name = 'train_all'
        mask_path = os.path.join(root, img_dir_name, 'LabelIds')
    else:
        raise BaseException("Instance Segmentation Not support")

    img_path = os.path.join(root, img_dir_name, 'Images')
    if quality != 'video':
        imgs = sorted([os.path.splitext(f)[0] for f in os.listdir(img_path)])
        msks = sorted([os.path.splitext(f)[0] for f in os.listdir(mask_path)])
        assert imgs == msks

    items = []
    c_items = os.listdir(img_path)
    if '.DS_Store' in c_items:
        c_items.remove('.DS_Store')

    for it in c_items:
        if quality == 'video':
            item = (os.path.join(img_path, it), os.path.join(img_path, it))
        else:
            item = (os.path.join(img_path, it),
                    os.path.join(mask_path, it.replace(".jpg", ".png")))
        items.append(item)
    return items

def make_dataset(root, quality, mode):
    """
    Create File List
    """
    assert (quality == 'semantic' and mode in ['train', 'val', 'trainval', 'test']), (quality, mode)
    img_dir_name = None
    if quality == 'semantic':
        if mode == 'train':
            img_dir_name = 'train_prepare'
        if mode == 'val':
            img_dir_name = 'train_prepare'
        if mode == 'trainval':
            img_dir_name = ['train_prepare', 'train_prepare']
        if mode == 'test':
            img_dir_name = 'test_prepare'
    else:
        raise BaseException("Instance Segmentation Not support")
    if not isinstance(img_dir_name, list):
        img_dir_name = [img_dir_name]
    items = []
    img_dir = img_dir_name[0]
    img_path = os.path.join(root, img_dir, 'images')
    mask_path = os.path.join(root, img_dir, 'labels_id')

    c_items = os.listdir(img_path)
    if '.DS_Store' in c_items:
        c_items.remove('.DS_Store')

    for it in c_items:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
        items.append(item)
    return items

class ISPRS2DColorEncoder:
  def __init__(self):
    # color table.
    self.clr_tab = self.createColorTable()
    # id table.
    id_tab = {}
    for k, v in self.clr_tab.items():
        id_tab[k] = self.clr2id(v)
    self.id_tab = id_tab

  def createColorTable(self):
    clr_tab = {}
    clr_tab['Impervious_surfaces'] = [255, 255, 255]
    clr_tab['Building'] = [0, 0, 255]
    clr_tab['Vegetation'] = [0, 255, 255]
    clr_tab['Tree'] = [0, 255, 0]
    clr_tab['Car'] = [255, 255, 0]
    clr_tab['Clutter'] = [255, 0, 0]
    return clr_tab

  def colorTable(self):
    return self.clr_tab

  def clr2id(self, clr):
    return clr[0]+clr[1]*255+clr[2]*255*255

  def id2clr(self, id):
    return [id%255, id//255%(255), id//(255*255)]

  def clr2name(self, clr):
      for name, cur_clr in self.clr_tab.items():
          if cur_clr == list(clr):
              return name
      return None

  #transform to uint8 integer label
  def transform(self,label, dtype=np.int32):
    height,width = label.shape[:2]
    # default value is index of clutter.
    newLabel = np.zeros((height, width), dtype=dtype)
    id_label = label.astype(np.int64)
    id_label = id_label[:,:,0]+id_label[:,:,1]*255+id_label[:,:,2]*255*255
    for tid,val in enumerate(self.id_tab.values()):
      mask = (id_label == val)
      newLabel[mask] = tid
    return newLabel

  #transform back to 3 channels uint8 label
  def inverse_transform(self, label):
    label_img = np.zeros(shape=(label.shape[0], label.shape[1],3),dtype=np.uint8)
    values = list(self.clr_tab.values())
    for tid,val in enumerate(values):
      mask = (label==tid)
      label_img[mask] = val
    return label_img

class Loader(BaseLoader):
    num_classes = 6
    ignore_label = 255
    clr_encoder = ISPRS2DColorEncoder()
    def __init__(self, mode, quality='semantic', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality,
                                     mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        root = cfg.DATASET.POTSDAM_DIR
        self.mode = mode
        self.fill_colormap()
        ######################################################################
        # Assemble image lists
        ######################################################################
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:
            splits = {'train': 'training',
                      'val': 'validation',
                      'trainval': 'trainval',
                      'test': 'testing'}
            split_name = splits[mode]
            #img_ext = 'jpg'
            #mask_ext = 'png'
            #img_root = os.path.join(root, split_name, 'images')
            #mask_root = os.path.join(root, split_name, 'labels')
            if mode is 'test':
                self.all_imgs = make_dataset(root, quality, mode)
            else:
                self.all_imgs = make_dataset(root, quality, mode)
        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    def fill_colormap(self):
        self.trainid_to_name = {}
        self.color_mapping = []
        for tid, (name, clr) in enumerate(self.clr_encoder.clr_tab.items()):
            self.trainid_to_name[tid] = name
            self.color_mapping = self.color_mapping+clr

    #Reload read_images function.
    def read_images(self, img_path, mask_path, mask_out=False):
        img = Image.open(img_path).convert('RGB')
        if self.mode is 'test' or mask_path is None or mask_path == '':
            w, h = img.size
            mask = np.zeros((h, w))
        else:
            mask = Image.open(mask_path)

        drop_out_mask = None
        if cfg.dump_with_subdir_level>0:
            dirs = os.path.dirname(img_path).split(os.path.sep)
            dir_name = os.path.join(*dirs[-cfg.dump_with_subdir_level:])
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            #img_name is used for dumping result, we replace 'Images' with 'Labels' for the output folder.
            img_name = os.path.join(dir_name, img_name).replace('images', 'labels')
        else:
            img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        if (mask_out):
            mask = self.drop_mask * mask

        mask = mask.copy()
        for k, v in self.id_to_trainid.items():
            binary_mask = (mask == k) #+ (gtCoarse == k)
            mask[binary_mask] = v

        mask = Image.fromarray(mask.astype(np.uint8))
        return img, mask, img_name