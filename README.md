# Spring_2021_DL_Project
Model Compression and Acceleration Analysis on Image Classification Task

Mingxi Chen (mc7805) Xuefei Zhou (xz2643)

## Channel Level Pruning

### Folder Structure
```
channel_level_pruning
├── models
│   └────── __init__.py
│   └────── channel_selection.py
│   └────── preresnet.py
│   └────── preresnet_flops.py   
│   └────── vgg.py    
│     
│── b.sh  
│     
│── flops.py       
│── inference.py      
│── throughput.py 
│     
│── main.py      
│    
│── resprune.py  
│  
└── vggprune.py
```
### Training/Pruning/Fine-tuning Platform
NYU Greene

### Command
```
srun --mem=26GB --time=10:00:00 --cpus-per-task 4 --gres=gpu:v100:1 --pty $SHELL

module load python/intel/3.8.6

==========================================================================

                                 VGG

==========================================================================

python3 main.py --dataset cifar10 --arch vgg --depth 19

python3 inference.py --inference log_vgg_baseline/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 1

python3 throughput.py --inference log_vgg_baseline/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 10000

python3 flops.py  --arch vgg --depth 19

==========================================================================

                           VGG_pruned_70

==========================================================================

python3 main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save log

python3 vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model log_vgg_slim/model_best.pth.tar --save vgg_pruned

# Configuration:
# [38, 64, 'M', 127, 128, 'M', 253, 250, 217, 172, 'M', 124, 45, 40, 41, 'M', 20, 22, 39, 71]

./b.sh &

python3 main.py --refine vgg_pruned/pruned.pth.tar --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save log_vgg_pruned

python3 inference.py -pr --inference log_vgg_pruned/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 1

python3 throughput.py -pr --inference log_vgg_pruned/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 10000

python3 flops.py -pr --arch vgg --depth 19

==========================================================================

                           VGG_pruned_60

==========================================================================

python3 vggprune.py --dataset cifar10 --depth 19 --percent 0.6 --model log_vgg_slim/model_best.pth.tar --save vgg_pruned_60

# Configuration:
# [39, 64, 'M', 127, 128, 'M', 253, 250, 224, 179, 'M', 135, 76, 85, 109, 'M', 130, 149, 168, 85]

./b.sh &

python3 main.py --refine vgg_pruned_60/pruned.pth.tar --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save log_vgg_pruned_60

python3 inference.py -pr --inference log_vgg_pruned_60/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 1

python3 throughput.py -pr --inference log_vgg_pruned_60/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 10000
Final Throughput: 34914.32681825829

python3 flops.py -pr --arch vgg --depth 19

==========================================================================

                           VGG_pruned_65

==========================================================================

python3 vggprune.py --dataset cifar10 --depth 19 --percent 0.65 --model log_vgg_slim/model_best.pth.tar --save vgg_pruned_65

# Configuration:
# [39, 64, 'M', 127, 128, 'M', 253, 250, 224, 177, 'M', 132, 64, 76, 89, 'M', 51, 74, 101, 77]

python3 main.py --refine vgg_pruned_65/pruned.pth.tar --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save log_vgg_pruned_65

python3 inference.py -pr --inference log_vgg_pruned_65/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 1

python3 throughput.py -pr --inference log_vgg_pruned_65/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 10000

python3 flops.py -pr --arch vgg --depth 19

==========================================================================

                           VGG_pruned_75

==========================================================================

python3 vggprune.py --dataset cifar10 --depth 19 --percent 0.75 --model log_vgg_slim/model_best.pth.tar --save vgg_pruned_75

# Configuration:
# [35, 64, 'M', 127, 128, 'M', 253, 249, 193, 125, 'M', 75, 21, 6, 2, 'M', 8, 9, 13, 67]

python3 main.py --refine vgg_pruned_75/pruned.pth.tar --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save log_vgg_pruned_75

python3 inference.py -pr --inference log_vgg_pruned_75/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 1

python3 throughput.py -pr --inference log_vgg_pruned_75/model_best.pth.tar --dataset cifar10 --arch vgg --depth 19 --test-batch-size 10000

python3 flops.py -pr --arch vgg --depth 19

==========================================================================

                               ResNet

==========================================================================
python3 main.py --dataset cifar10 --arch resnet --depth 50 --save log_resnet_baseline2

python3 inference.py --inference log_resnet_baseline2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 1

python3 throughput.py --inference log_resnet_baseline2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 500

python3 flops.py --arch resnet --depth 50

==========================================================================

                           VGG_pruned_45

==========================================================================

python3 main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 50 --save log_resnet_slim2


python3 resprune.py --dataset cifar10 --depth 50 --percent 0.45 --model log_resnet_slim2/model_best.pth.tar --save resnet_pruned_45_2

#Configuration:
# [50, 59, 64, 158, 64, 61, 116, 62, 64, 239, 128, 128, 369, 128, 128, 334, 128, 128, 330, 127, 128, 477, 256, 256, 500, 255, 256, 626, 256, 256, 653, 256, 256, 734, 256, 256, 717, 256, 256, 146, 184, 328, 28, 43, 82, 41, 52, 150, 1670]

python3 main.py --refine resnet_pruned_45_2/pruned.pth.tar --dataset cifar10 --arch resnet --depth 50 --epochs 160 --save log_resnet_pruned_45_2

python3 inference.py -pr --inference log_resnet_pruned_45_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 1

python3 throughput.py -pr --inference log_resnet_pruned_45_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 500

python3 flops.py -pr --arch resnet_flops --depth 50

==========================================================================

                           VGG_pruned_45

==========================================================================

python3 resprune.py --dataset cifar10 --depth 50 --percent 0.5 --model log_resnet_slim2/model_best.pth.tar --save resnet_pruned_50_2

# Configuration:
# [49, 58, 64, 146, 64, 61, 102, 58, 64, 230, 128, 128, 283, 128, 128, 290, 127, 128, 283, 127, 128, 458, 256, 256, 354, 253, 256, 501, 255, 256, 556, 256, 256, 685, 256, 256, 664, 256, 256, 87, 90, 270, 15, 11, 68, 20, 20, 119, 1599]

python3 main.py --refine resnet_pruned_50_2/pruned.pth.tar --dataset cifar10 --arch resnet --depth 50 --epochs 160 --save log_resnet_pruned_50_2

python3 inference.py -pr --inference log_resnet_pruned_50_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 1

python3 throughput.py -pr --inference log_resnet_pruned_50_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 500

python3 flops.py -pr --arch resnet_flops --depth 50

==========================================================================

                           VGG_pruned_60

==========================================================================

python3 resprune.py --dataset cifar10 --depth 50 --percent 0.6 --model log_resnet_slim2/model_best.pth.tar --save resnet_pruned_60_2

# Configuration:
# [44, 56, 64, 100, 62, 61, 72, 51, 64, 182, 125, 128, 120, 121, 128, 138, 118, 127, 177, 126, 128, 418, 254, 256, 92, 191, 253, 226, 243, 256, 347, 250, 254, 545, 256, 254, 543, 254, 255, 20, 12, 175, 7, 3, 41, 4, 4, 77, 1405]

python3 main.py --refine resnet_pruned_60_2/pruned.pth.tar --dataset cifar10 --arch resnet --depth 50 --epochs 160 --save log_resnet_pruned_60_2

python3 inference.py -pr --inference log_resnet_pruned_60_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 1

python3 throughput.py -pr --inference log_resnet_pruned_60_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 500

python3 flops.py -pr --arch resnet_flops --depth 50

==========================================================================

                           VGG_pruned_60

==========================================================================

python3 resprune.py --dataset cifar10 --depth 50 --percent 0.55 --model log_resnet_slim2/model_best.pth.tar --save resnet_pruned_55_2

# Configuration:
# [47, 57, 64, 123, 63, 61, 88, 56, 64, 205, 126, 128, 200, 127, 128, 216, 123, 128, 231, 127, 128, 445, 255, 256, 202, 233, 256, 362, 253, 256, 450, 255, 256, 621, 256, 256, 608, 255, 255, 50, 32, 230, 8, 4, 51, 8, 12, 94, 1514]

python3 main.py --refine resnet_pruned_55_2/pruned.pth.tar --dataset cifar10 --arch resnet --depth 50 --epochs 160 --save log_resnet_pruned_55_2

python3 inference.py -pr --inference log_resnet_pruned_55_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 1

python3 throughput.py -pr --inference log_resnet_pruned_55_2/model_best.pth.tar --dataset cifar10 --arch resnet --depth 50 --test-batch-size 500

python3 flops.py -pr --arch resnet_flops --depth 50
```