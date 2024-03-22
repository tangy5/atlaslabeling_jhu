import os
import numpy as np
import nibabel as nib

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

def align_masks_with_ct(outdir, ct_file, masks_dir):
    # Load CT image to get header and affine information
    ct_img = nib.load(ct_file)
    ct_header = ct_img.header
    ct_affine = ct_img.affine

    # Iterate over each mask file in the directory
#     for root, dirs, files in os.walk(masks_dir):
#         for file in files:
    if masks_dir.endswith('.nii.gz'):
        # Load mask image
        mask_img = nib.load(masks_dir)
        mask_data = mask_img.get_fdata()

        # Create a separate mask file for each label
        for label, name in class_map_part_ribs.items():
            binary_mask = np.where(mask_data == label, 1, 0).astype(np.uint8)
            # Create a NIfTI object for saving
            mask_nifti = nib.Nifti1Image(binary_mask, affine=ct_affine, header=ct_header)
            # Save the binary mask image with a new filename
            mask_save_path = os.path.join(outdir, f"{name}.nii.gz")
            nib.save(mask_nifti, mask_save_path)
            print(f"Saved binary mask for label {label} ({name}) as {mask_save_path}")

if __name__ == "__main__":
    # Define your root directory containing subject directories
    root_dir = '/RW/2024/MICCAI24/jhu/BagofTricksNeedPseudoLabel/'
    # Iterate over each subject directory
    for subject_dir in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject_dir)
        if os.path.isdir(subject_path):
            ct_file = os.path.join(subject_path, 'ct.nii.gz')
            masks_dir = os.path.join(subject_path, 'segmentations', "all_vertebrae.nii.gz")
            out_dir = os.path.join(subject_path, 'segmentations_final')
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            if os.path.exists(ct_file) and os.path.exists(masks_dir):
                align_masks_with_ct(out_dir, ct_file, masks_dir)
            else:
                print(f"CT file or masks directory not found for {subject_dir}")
