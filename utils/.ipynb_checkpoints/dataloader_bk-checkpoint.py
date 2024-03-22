from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import os


import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import SimpleITK as sitk

sys.path.append("..") 
# from utils.utils import get_key

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class_map_part_cardiac = {
    1: "esophagus",
    2: "trachea",
    3: "heart_myocardium",
    4: "heart_atrium_left",
    5: "heart_ventricle_left",
    6: "heart_atrium_right",
    7: "heart_ventricle_right",
    8: "pulmonary_artery",
    9: "brain",
    10: "iliac_artery_left",
    11: "iliac_artery_right",
    12: "iliac_vena_left",
    13: "iliac_vena_right",
    14: "small_bowel",
    15: "duodenum",
    16: "colon",
    17: "urinary_bladder",
    18: "face"
    }

class_map_part_organs = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right"
    }

class_map_part_vertebrae = {
    1: "vertebrae_L5",
    2: "vertebrae_L4",
    3: "vertebrae_L3",
    4: "vertebrae_L2",
    5: "vertebrae_L1",
    6: "vertebrae_T12",
    7: "vertebrae_T11",
    8: "vertebrae_T10",
    9: "vertebrae_T9",
    10: "vertebrae_T8",
    11: "vertebrae_T7",
    12: "vertebrae_T6",
    13: "vertebrae_T5",
    14: "vertebrae_T4",
    15: "vertebrae_T3",
    16: "vertebrae_T2",
    17: "vertebrae_T1",
    18: "vertebrae_C7",
    19: "vertebrae_C6",
    20: "vertebrae_C5",
    21: "vertebrae_C4",
    22: "vertebrae_C3",
    23: "vertebrae_C2",
    24: "vertebrae_C1"
    }

class_map_part_muscles = {
    1: "humerus_left",
    2: "humerus_right",
    3: "scapula_left",
    4: "scapula_right",
    5: "clavicula_left",
    6: "clavicula_right",
    7: "femur_left",
    8: "femur_right",
    9: "hip_left",
    10: "hip_right",
    11: "sacrum",
    12: "gluteus_maximus_left",
    13: "gluteus_maximus_right",
    14: "gluteus_medius_left",
    15: "gluteus_medius_right",
    16: "gluteus_minimus_left",
    17: "gluteus_minimus_right",
    18: "autochthon_left",
    19: "autochthon_right",
    20: "iliopsoas_left",
    21: "iliopsoas_right"
    }

class_map_part_ribs = {
    1: 'rib_left_1', 
    2: 'rib_left_2', 
    3: 'rib_left_3', 
    4: 'rib_left_4', 
    5: 'rib_left_5', 
    6: 'rib_left_6', 
    7: 'rib_left_7', 
    8: 'rib_left_8', 
    9: 'rib_left_9', 
    10: 'rib_left_10', 
    11: 'rib_left_11', 
    12: 'rib_left_12', 
    13: 'rib_right_1', 
    14: 'rib_right_2', 
    15: 'rib_right_3', 
    16: 'rib_right_4', 
    17: 'rib_right_5', 
    18: 'rib_right_6', 
    19: 'rib_right_7', 
    20: 'rib_right_8', 
    21: 'rib_right_9', 
    22: 'rib_right_10', 
    23: 'rib_right_11', 
    24: 'rib_right_12'
    }

taskmap_set = {
    'cardiac': class_map_part_cardiac,
    'organs': class_map_part_organs,
    'vertebrae': class_map_part_vertebrae,
    'muscles': class_map_part_muscles,
    'ribs': class_map_part_ribs,
}


class LoadImaged_totoalseg(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        map_type,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = True,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting
        self.map_type = map_type


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                data = self._loader(d[key], reader)
            except:
                print(d['name'])
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
#         d['label'], d['label_meta_dict'] = self.label_transfer(d['label'], self.map_type, d['image'].shape)
        
        return d

    def label_transfer(self, lbl_dir, map_type, shape):
        organ_map = totalseg_taskmap_set[map_type]
        organ_lbl = np.zeros(shape)
        for index, organ in organ_map.items():
            array, mata_infomation = self._loader(lbl_dir + organ + '.nii.gz')
            organ_lbl[array == 1] = index
        
        return organ_lbl, mata_infomation

def get_loader(args):
    test_transforms = Compose(
        [
#             LoadImaged_totoalseg(keys=["image"], map_type=args.map_type),
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), 
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )
    
    
    ## test dict part
    test_img = []
    test_lbl = []
    test_name = []
    for line in open(args.data_txt_path + 'bagoftricks_200.txt'):
        name = line.strip().split('\t')[0]
        test_img.append(args.dataset_path + name + '/ct.nii.gz')
        test_lbl.append(args.dataset_path + name + '/segmentations/')
        test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(test_img, test_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))
    
    test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    return test_loader, test_transforms


if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()
