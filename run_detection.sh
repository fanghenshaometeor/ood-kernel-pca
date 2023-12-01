# ======== CIFAR10-R18 ========

python CoP_CoRP_c10.py \
 --arch R18 --in_data CIFAR10 --out_datasets SVHN LSUN iSUN Texture places365 \
 --method CoP --exp_var_ratio 0.85 # CoP, standard training

python CoP_CoRP_c10.py \
 --arch R18 --in_data CIFAR10 --out_datasets SVHN LSUN iSUN Texture places365 \
 --method CoP --exp_var_ratio 0.85 --supcon  # CoP, supervised constrastive learning

python CoP_CoRP_c10.py \
 --arch R18 --in_data CIFAR10 --out_datasets SVHN LSUN iSUN Texture places365 \
 --method CoRP --exp_var_ratio 0.8 --gamma 2 --M 2048 # CoRP, standard training

python CoP_CoRP_c10.py \
 --arch R18 --in_data CIFAR10 --out_datasets SVHN LSUN iSUN Texture places365 \
 --method CoRP --exp_var_ratio 0.7 --gamma 1 --M 2048 --supcon # CoRP, supervised constrastive learning



# ======== ImageNet-R50 ========

python CoP_CoRP_ImgNet.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoP --exp_var_ratio 0.99 # CoP, standard training

python CoP_CoRP_ImgNet.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoP --exp_var_ratio 0.7 --supcon  # CoP, supervised constrastive learning

python CoP_CoRP_ImgNet.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoRP --exp_var_ratio 0.5 --gamma 3 --M 4096 # CoRP, standard training

python CoP_CoRP_ImgNet.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method CoRP --exp_var_ratio 0.5 --gamma 1 --M 4096 --supcon # CoRP, supervised constrastive learning
