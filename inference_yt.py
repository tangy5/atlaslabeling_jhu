import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import nibabel as nib

import warnings
warnings.filterwarnings("ignore")

from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

from utils.dataloader_bk import get_loader,taskmap_set
from utils.postprocess import TEMPLATE, organ_post_process, invert_transform
from model.SwinUNETR import SwinUNETR

def validation(model, ValLoader, test_transforms, args):
    model.eval()
    selected_class_map = taskmap_set[args.map_type]

    output_directory = os.path.join('out', args.log_name)
    os.makedirs(output_directory, exist_ok=True)
    count = 0
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name = batch["image"].to(args.device), batch["name"]

        name = name.item() if isinstance(name, torch.Tensor) else name  # Convert to Python str if it's a Tensor
        original_affine = batch["image_meta_dict"]["affine"][0].numpy()

        with torch.no_grad():
            print("Image: {}, shape: {}".format(name[0], image.shape))
            val_outputs = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian',device=torch.device("cpu"))
            val_outputs = val_outputs.sigmoid()
#             hard_val_outputs = (val_outputs>0.5).float()
            print(val_outputs.shape)
            hard_val_outputs = torch.argmax(val_outputs, dim=1).unsqueeze(1)
            print(hard_val_outputs.shape)
            print(np.unique(hard_val_outputs))

        
#         organ_list_all = TEMPLATE['totalseg'] # post processing all organ
#         pred_hard_post, _ = organ_post_process(hard_val_outputs.cpu().numpy(), organ_list_all,args)
#         pred_hard_post = torch.tensor(pred_hard_post)
        
        batch["pred"] = hard_val_outputs
        batch = invert_transform('pred', batch, test_transforms)
        pred = batch[0]['pred'].cpu().numpy()[0]
        print(pred.shape)
        
        file_path_pattern = os.path.join(args.dataset_path, name[0], "segmentations", "all_vertebraes_real.nii.gz")
        nib.save(
                nib.Nifti1Image(pred.astype(np.uint8), original_affine), file_path_pattern
        )
#         for k in range(1, args.num_class):
#             class_name = selected_class_map[k]
#             pred_class = pred[k]
#             # Construct the file path based on the class name and name
#             file_path_pattern = os.path.join(args.dataset_path, name[0], "segmentations", f"{class_name}.nii.gz")
#             nib.save(
#                     nib.Nifti1Image(pred_class.astype(np.uint8), original_affine), file_path_pattern
#             )
        count += 1
        print("[{}/{}] Saved {}".format(count,len(ValLoader),name[0]))

#         organ_list_all = TEMPLATE['totalseg'] # post processing all organ

#         organ_seg_save_path = output_directory+'/pred'
#         os.makedirs(organ_seg_save_path, exist_ok=True)

#         final_pred = np.zeros(pred.shape[1:])
#         for i in selected_class_map.keys():
#             final_pred[pred[i-1]==1] = i

#         nib.save(
#                 nib.Nifti1Image(final_pred.astype(np.uint8), original_affine), os.path.join(organ_seg_save_path, f'{name[0]}_pred.nii.gz')
#         )


def process(args):
    rank = 0
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
#     if args.model_backbone == 'segresnet':
#         model = SegResNet(
#                     blocks_down=[1, 2, 2, 4],
#                     blocks_up=[1, 1, 1],
#                     init_filters=16,
#                     in_channels=1,
#                     out_channels=32,
#                     dropout_prob=0.0,
#                     )
#     else:
#         model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
#                         in_channels=1,
#                         out_channels=32,
#                         backbone=args.model_backbone,
#                         encoding='word_embedding'
#                         )

    if args.model_backbone == 'swinunetr':
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=args.num_class,
                    feature_size=48,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    dropout_path_rate=0.0,
                    use_checkpoint=False
                    )
        store_dict = model.state_dict()
        model_dict = torch.load(args.pretrain)['net']
        store_dict = model.state_dict()
        amount = 0
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if new_key in store_dict.keys():
                store_dict[new_key] = model_dict[key]   
                amount += 1
        model.load_state_dict(store_dict)
        print(amount, len(store_dict.keys()))
        print(f'Load Swin UNETR transfer learning weights')
        
    model.to(args.device)

    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.pretrain)
    load_dict = checkpoint['net']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    model.load_state_dict(store_dict)
    print('Use pretrained weights')
    model.cuda()
    torch.backends.cudnn.benchmark = True

    test_loader, test_transforms = get_loader(args)

    validation(model, test_loader, test_transforms, args)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")

    ## logging
    parser.add_argument('--log_name', default='vertebrae_r2', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--pretrain', default='./models/best_model_vertebrae.pth', 
                        help='The path of pretrain model')
    parser.add_argument('--map_type', default='vertebrae', help='sample number in each ct, vertebrae, ribs')

    # change here
    parser.add_argument('--data_txt_path', default='./', help='data txt path')
    parser.add_argument('--dataset_path', default='/RW/2024/MICCAI24/jhu/BagofTricksNeedPseudoLabel/', help='dataset path')
    

    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-250, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_class', default=25, type=int, help='class num')

    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')
    parser.add_argument('--model_backbone', default='swinunetr', help='model backbone, also avaliable for swinunetr')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
