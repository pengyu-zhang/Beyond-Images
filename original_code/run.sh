#!/bin/bash
# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00


source activate data311


# python /gpfs/work5/0/prjs1145/code/beyond_images/01_original_images/1.1_wikidata_url.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/01_original_images/1.2_rename_img.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/01_original_images/1.3_error_and_fail_log.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/01_original_images/1.4_match.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/01_original_images/1.5_DB15K_download-images.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/01_original_images/1.6_DB15K_rename_img.py


# python /gpfs/work5/0/prjs1145/code/beyond_images/02_new_images/2.1_new_img.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/02_new_images/2.2_new_img_mkgy.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/02_new_images/2.3_db15k_links.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/02_new_images/2.4_db15k_wiki_links.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/02_new_images/2.5_db15k_new_img.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/02_new_images/2.6_new_img_MKG-W_rename.py


# source activate blip2311
# python /gpfs/work5/0/prjs1145/code/beyond_images/04_img_to_text_blip2/4.1_MKG-W_img_to_text_blip2.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/04_img_to_text_blip2/4.2_MKG-Y_img_to_text_blip2.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/04_img_to_text_blip2/4.3_DB15K_img_to_text_blip2.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/04_img_to_text_blip2/4.4_MKG-W_img_summary.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/04_img_to_text_blip2/4.5_MKG-Y_img_summary.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/04_img_to_text_blip2/4.6_DB15K_img_summary.py


python /gpfs/work5/0/prjs1145/code/beyond_images/05_text_to_vector/5.1_check.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/05_text_to_vector/5.2_json.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/05_text_to_vector/5.3_merge_json.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/05_text_to_vector/5.4_read_pth_and_h5.py
# python /gpfs/work5/0/prjs1145/code/beyond_images/05_text_to_vector/5.5_text_to_vector.py