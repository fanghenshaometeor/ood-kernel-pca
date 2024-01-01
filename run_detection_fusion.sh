# ======== FUSION with MSP, Energy, ReAct, BATS

# ---- CoP+MSP/Energy/ReAct/BATS on R50 and MNet
for fmethod in MSP Energy ReAct BATS
do

CUDA_VISIBLE_DEVICES=0 python CoP_CoRP_fusion.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoP --exp_var_ratio 0.9 \
 --fmethod ${fmethod}

CUDA_VISIBLE_DEVICES=0 python CoP_CoRP_fusion.py \
 --arch MNet --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoP --exp_var_ratio 0.6 \
 --fmethod ${fmethod}

done


# ---- CoRP+MSP/Energy on R50 and MNet
for fmethod in MSP Energy
do

CUDA_VISIBLE_DEVICES=0 python CoP_CoRP_fusion.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoRP --exp_var_ratio 0.5 --gamma 2 --M 4096 \
 --fmethod ${fmethod}

CUDA_VISIBLE_DEVICES=0 python CoP_CoRP_fusion.py \
 --arch MNet --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoRP --exp_var_ratio 0.5 --gamma 1 --M 2560 \
 --fmethod ${fmethod}

done

# ---- CoRP+ReAct/BATS on R50 and MNet
for fmethod in ReAct BATS
do

CUDA_VISIBLE_DEVICES=0 python CoP_CoRP_fusion.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoRP --exp_var_ratio 0.8 --gamma 0.5 --M 4096 \
 --fmethod ${fmethod}

CUDA_VISIBLE_DEVICES=0 python CoP_CoRP_fusion.py \
 --arch MNet --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoRP --exp_var_ratio 0.6 --gamma 1 --M 2560 \
 --fmethod ${fmethod}

done

