
import os
import zipfile
import glob

def zip_directory_exclude_segmentations(root_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Temporarily store dirs to remove to avoid modifying the list during iteration
            dirs_to_remove = []
            
            for dir in dirs:
                if dir == 'segmentations':
                    # Check if the number of nii.gz files in segmentations is not 48
                    subject_dir = os.path.basename(os.path.normpath(root))
                    segmentations_path = os.path.join(root, dir)
                    dirs_to_remove.append(dir)
                    
                if dir == 'segmentations_final':
                    # Check if the number of nii.gz files in segmentations is not 48
                    subject_dir = os.path.basename(os.path.normpath(root))
                    segmentations_path = os.path.join(root, dir)
                    
                    nii_gz_files = glob.glob(os.path.join(segmentations_path, '*.nii.gz'))
                    if len(nii_gz_files) != 48:
                        print(f"Subject with incorrect file count: {subject_dir}")
                    
            # Remove 'segmentations' from dirs to prevent it from being walked
            for dir_to_remove in dirs_to_remove:
                dirs.remove(dir_to_remove)
            
                
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=root_dir)
                
                # Rename 'segmentations_final' to 'segmentations' in the zip path
                if 'segmentations_final' in arcname:
                    arcname = arcname.replace('segmentations_final', 'segmentations')
                zipf.write(file_path, arcname)


root_dir = '/RW/2024/MICCAI24/jhu/BagofTricksNeedPseudoLabel'  # Replace with your actual root_dir path
output_zip = '/RW/2024/MICCAI24/jhu/BagofTricksPseudoLabel.zip'  # Replace with your desired output zip file path

zip_directory_exclude_segmentations(root_dir, output_zip)
