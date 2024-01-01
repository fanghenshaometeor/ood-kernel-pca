# -------- extract features from c10-r18 (standard training and supervised contrastive learning)
for out_data in SVHN LSUN iSUN Texture places365
do

CUDA_VISIBLE_DEVICES=0 python feat_extract.py \
 --arch R18 --in_data CIFAR10 --out_data ${out_data} \
 --model_path ./save/CIFAR10/R18/ce/checkpoint_100.pth.tar

CUDA_VISIBLE_DEVICES=0 python feat_extract.py \
 --arch R18 --in_data CIFAR10 --out_data ${out_data} \
 --model_path ./save/CIFAR10/R18/supcon/checkpoint_500.pth.tar --supcon # supervised constrastive learning

done

# -------- extract features from imagenet-r50/mnet 
# ---- pre-trained (r50,mnet) and supervised contrastive learning (r50)
for out_data in iNaturalist SUN Places Texture
do

CUDA_VISIBLE_DEVICES=0 python feat_extract_largescale.py \
 --arch R50 --in_data ImageNet --out_data ${out_data} --batch_size 128

CUDA_VISIBLE_DEVICES=0 python feat_extract_largescale.py \
 --arch R50 --in_data ImageNet --out_data ${out_data} --batch_size 128 \
 --model_path ./save/ImageNet/R50/supcon/supcon.pth --supcon

CUDA_VISIBLE_DEVICES=0 python feat_extract_largescale.py \
 --arch MNet --in_data ImageNet --out_data ${out_data} --batch_size 128

done
