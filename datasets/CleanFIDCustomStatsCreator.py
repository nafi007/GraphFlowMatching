import os
from PIL import Image
from torchvision import transforms

# ############################################################ OLD ##########################
# # # Input directory with original images (e.g., 512x512)
# # input_dir = "/home/shahriar/data/afhq"  # Contains subfolders "train" and "val"
# # # Output directory for resized images
# # output_dir = "/home/shahriar/data/afhq_64" #"/path/to/afhq_64"

# # # Define splits and categories
# # splits = ["train", "val"]
# # categories = ["cat", "dog", "wild"]

# # # Define the resizing transform (matching your training pipeline)
# # resize_transform = transforms.Resize((64, 64))

# # for split in splits:
# #     for category in categories:
# #         in_folder = os.path.join(input_dir, split, category)
# #         out_folder = os.path.join(output_dir, split, category)
# #         os.makedirs(out_folder, exist_ok=True)

# #         for img_name in os.listdir(in_folder):
# #             if img_name.lower().endswith(".jpg"):
# #                 in_path = os.path.join(in_folder, img_name)
# #                 out_path = os.path.join(out_folder, img_name)
                
# #                 # Open, convert, and resize the image
# #                 with Image.open(in_path) as img:
# #                     img = img.convert("RGB")
# #                     img_resized = resize_transform(img)
# #                     img_resized.save(out_path)

# ############################################################ NEW ############################
# # Input directory with original images (e.g., 512x512)
# input_dir = "/data/shahriar/datasets/afhq_v2"  # Contains subfolders "train" and "val"
# # Output directory for resized images
# output_dir = "/data/shahriar/datasets/afhq_v2_256"  # "/path/to/afhq_64"

# # Define splits and categories
# splits = ["train", "test"]
# categories = ["cat", "dog", "wild"]

# # Define the resizing transform (matching your training pipeline)
# resize_transform = transforms.Resize((256, 256))

# for split in splits:
#     for category in categories:
#         in_folder = os.path.join(input_dir, split, category)
#         out_folder = os.path.join(output_dir, split, category)
#         os.makedirs(out_folder, exist_ok=True)

#         for img_name in os.listdir(in_folder):
#             if img_name.lower().endswith(".png"):
#                 in_path = os.path.join(in_folder, img_name)
#                 # Change the extension to .png
#                 out_path = os.path.join(out_folder, os.path.splitext(img_name)[0] + ".png")
                
#                 # Open, convert, and resize the image
#                 with Image.open(in_path) as img:
#                     img = img.convert("RGB")
#                     img_resized = resize_transform(img)
#                     img_resized.save(out_path)




# from cleanfid import fid

# fid.make_custom_stats("afhq_cat_256_train", "/data/shahriar/datasets/afhq_v2_256/train/cat",mode="clean")

# fid.make_custom_stats("afhq_cat_256_test", "/data/shahriar/datasets/afhq_v2_256/test/cat",mode="clean") #"legacy_pytorch"

# fid.make_custom_stats("afhq_cat_256_train_test_full", "/data/shahriar/datasets/afhq_v2_256/train_test_full/cat",mode="clean")

'''
For test split:
saving custom FID stats to /home/shahriar/miniconda3/envs/flowmatch/lib/python3.9/site-packages/cleanfid/stats/afhq_cat_256_test_clean_custom_na.npz
saving custom KID stats to /home/shahriar/miniconda3/envs/flowmatch/lib/python3.9/site-packages/cleanfid/stats/afhq_cat_256_test_clean_custom_na_kid.npz

For train split:
saving custom FID stats to /home/shahriar/miniconda3/envs/flowmatch/lib/python3.9/site-packages/cleanfid/stats/afhq_cat_256_train_clean_custom_na.npz
saving custom KID stats to /home/shahriar/miniconda3/envs/flowmatch/lib/python3.9/site-packages/cleanfid/stats/afhq_cat_256_train_clean_custom_na_kid.npz

For train+test split:
saving custom FID stats to /home/shahriar/miniconda3/envs/flowmatch/lib/python3.9/site-packages/cleanfid/stats/afhq_cat_256_train_test_full_clean_custom_na.npz
saving custom KID stats to /home/shahriar/miniconda3/envs/flowmatch/lib/python3.9/site-packages/cleanfid/stats/afhq_cat_256_train_test_full_clean_custom_na_kid.npz

'''

# score = fid.compute_fid("folder_fake", dataset_name="afhq_cat_64",
#           mode="legacy_pytorch", dataset_split="custom")

# score = fid.compute_fid("/home/shahriar/data/afhq_64/train/cat", dataset_name="afhq_cat_64",dataset_split='custom',
#                     mode="legacy_pytorch")


# print("")


# the score of the train split for afhq cat 64 to itself is 1.089793607889078e-05


# import os
# import numpy as np
# from cleanfid import fid

# # Set paths
# original_dir = "/home/shahriar/data/afhq_64/train/cat"
# first_half_dir = "/home/shahriar/data/afhq_64/train/cat_first_half"
# second_half_dir = "/home/shahriar/data/afhq_64/train/cat_second_half"

# # Create split directories
# os.makedirs(first_half_dir, exist_ok=True)
# os.makedirs(second_half_dir, exist_ok=True)

# # Get sorted list of images
# all_images = sorted([f for f in os.listdir(original_dir) if f.endswith(".jpg")])

# # Split into halves
# split_idx = len(all_images) // 2
# first_half = all_images[:split_idx]
# second_half = all_images[split_idx:]

# # Create symbolic links to avoid copying files
# for fname in first_half:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(first_half_dir, fname)
#     if not os.path.exists(dst):
#         os.symlink(src, dst)

# for fname in second_half:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(second_half_dir, fname)
#     if not os.path.exists(dst):
#         os.symlink(src, dst)

# # Compute FID between the two halves
# score = fid.compute_fid(first_half_dir,second_half_dir,mode="legacy_pytorch")

# print(f"FID between first half and second half: {score:.2f}")

# # answer is 5.51