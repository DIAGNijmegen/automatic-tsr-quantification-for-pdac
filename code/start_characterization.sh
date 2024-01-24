#!/bin/bash

image_to_process=$1

filename=$(basename -- "$image_to_process")

output_tb=/home/user/data/tmp_masks/${filename%.*}_tissue.tif
output_epi=/home/user/data/tmp_masks/${filename%.*}_epithelium.tif
output_tum=/home/user/data/tmp_masks/${filename%.*}_tumor.tif
convex_hulls_path=/home/user/data/concave_hull_masks

#MODEL TB PATH
model_tb_path="/home/user/source/models/tb/playground_soft-cloud-137_best_model.pt"
#model_tb_path="/data/pathology/users/pierpaolo/Pierpaolo/dockers/epithelium_segmentation_gc/models/tb/playground_soft-cloud-137_best_model.pt"
#MODEL EPITHELIUM PATH
model_epi_path="/home/user/source/models/epithelium/best_models"
#model_epi_path="/data/pathology/users/pierpaolo/Pierpaolo/dockers/epithelium_segmentation_gc/models/epithelium/best_models"



extension="${image_to_process##*.}"
echo "Extension: $extension"

echo $image_to_process

if [[ "$extension" != "tif" ]] 
    then
    echo "Converting: $filename"
    output_tif_path=/home/user/data/tmp_masks/${filename%.*}_converted.tif
    python3 /home/user/source/code/convert.py --input_path "$image_to_process" \
                                                --output_dir "/home/user/tmp_input" \
                                                --ext "$extension" 
fi

echo "starting tissue segmentation"
python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
                                            --input_wsi_path="/home/user/data/tmp_input/*.tif" \
                                            --output_wsi_path="/home/user/data/tissue_masks/{image}_tissue.tif" \
                                            --model_path=${model_tb_path} \
                                            --read_spacing=4.0 \
                                            --write_spacing=4.0 \
                                            --tile_size=512 \
                                            --readers=20 \
                                            --writers=20 \
                                            --batch_size=90 \
                                            --gpu_count=1 \
                                            --axes_order='cwh' \
                                            --custom_processor="torch_processor" \
                                            --reconstruction_information="[[0,0,0,0],[1,1],[96,96,96,96]]" \
                                            --quantize 



echo "starting epithelium segmentation"
python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
                                                --input_wsi_path="/home/user/data/tmp_input/*.tif" \
                                                --output_wsi_path="/home/user/data/tmp_masks/{image}_epithelium.tif" \
                                                --model_path="${model_epi_path}" \
                                                --read_spacing=1.0 \
                                                --write_spacing=1.0 \
                                                --mask_wsi_path="/home/user/data/tissue_masks/{image}_tissue.tif" \
                                                --mask_spacing=4.0 \
                                                --mask_class=2 \
                                                --tile_size=512 \
                                                --readers=20 \
                                                --writers=20 \
                                                --batch_size=70 \
                                                --gpu_count=1 \
                                                --axes_order='cwh' \
                                                --custom_processor="torch_processor_pier" \
                                                --reconstruction_information="[[0,0,0,0],[1,1],[100,100,100,100]]" \
                                                --quantize  



echo "starting convex hulls"
python3 /home/user/source/code/concave_hull.py \
                                --input_path "/home/user/data/tmp_masks" \
                                --output_dir "${convex_hulls_path}" \
                                --cls 3 \
                                --alpha 0.038 \
                                --min_size 290 \
                                --input_level 4 \
                                --output_level 2 \
                                --level_offset 4

echo "starting converting annotation"
python3 /home/user/source/pathology-common/scripts/convertannotations.py \
    -i "/home/user/data/tmp_input/*.tif"\
    -a "${convex_hulls_path}/{image}_epithelium.xml"\
    -m "${output_tum}" \
    -s 2.0 

echo "DONE"
