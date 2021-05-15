# Spring2021\_DL\_Project
Model Compression and Acceleration Analysis on Image Classification Task

Mingxi Chen (mc7805) Xuefei Zhou (xz2643)


## Folder Structure

```
├── code
│     |----- main.py # Handle HTTP requests
│     |----- inference.py # According to different request, choose corresponding model to predict user image 
│     |----- templates
                 |------- index.html
│     |----- ...
│   
│── Dockerfile # Used to build image with cuda (DEFAULT OPTION)
│
│── Dockerfile-cpu # Used to build image without cuda                
│ 
├── README.md
```
## Commands

### Step one: 
### Download our trained models
```
cd code

# To save trained original ResNet50
# Download from https://drive.google.com/file/d/1QXfytkqN5rIkP0rprC5WeYDqO6xIHO3z/view?usp=sharing
mkdir resnet
mv [YOUR_DOWNLOAD_PATH]/model_best.pth.tar resnet/

# To save trained original VGG19
# Download from https://drive.google.com/file/d/1QRg4Ph3yP-POPRj-_LUgPDCFmJOL_7F5/view?usp=sharing
mkdir vgg
mv [YOUR_DOWNLOAD_PATH]/model_best.pth.tar vgg/

# To save trained channel-level pruned ResNet50
# Download from https://drive.google.com/file/d/1C7clSb779qpWcmGm56TQQ2AXNzDtmkLE/view?usp=sharing
mkdir resnet_pruned
mv [YOUR_DOWNLOAD_PATH]/model_best.pth.tar resnet_pruned/

# To save trained channel-level pruned VGG19
# Download from https://drive.google.com/file/d/1N-diRASGDNFJGi24P3klMQYgB4djAkB8/view?usp=sharing
mkdir vgg_pruned
mv [YOUR_DOWNLOAD_PATH]/model_best.pth.tar vgg_pruned/

# To save trained filter-level pruned ResNet50
# Download from https://drive.google.com/file/d/1dV2dbXDY8271CPf5gWRrwZ_otqbUkHys/view?usp=sharing
mkdir resnet_trained_models
mv [YOUR_DOWNLOAD_PATH]/check_point_retrain.pth resnet_trained_models/

# To save trained filter-level pruned VGG19
# Download from https://drive.google.com/file/d/14tAFzSgqWpQGMoc-9s2Scs-H4jlT2kYf/view?usp=sharing
mkdir vgg_trained_models
mv [YOUR_DOWNLOAD_PATH]/check_point.pth vgg_trained_models/

```

### Build docker image and push it to Google Cloud Registry

```
docker build -t dl_project:v1 .

docker tag dl_project:v5 us.gcr.io/[YOUR_PROJECT_ID]/dl_project:v1

gcloud auth activate-service-account --key-file [YOUR_GOOGLE_CONTAINER_REGISTRY_KEY_FILE]

docker login -u _json_key -p "$(cat key_file.json)" https://gcr.io

docker push us.gcr.io/[YOUR_PROJECT_ID]/dl_project:v1

```
### Step two:
### Create VM instance
```
1. Login to GCP
2. Go to Console
3. Go to Compute Engine
4. Create New VM Instance
5. Select V100 GPU
6. Change to 50GB disk
7. Allow HTTP/HTTPS traffic
```
### Step three:
### Add new firewall rule for VM instance

```
1. Go to VPC Network
2. Go to Firewall
3. Create Firewall Rule
4. Name your new rule
5. Source IP ranges: 0.0.0.0/0
6. Specified protocols and ports: tcp 5000
```


### Step four:
### Run container in VM instance
```
# First ssh to your VM instance

cos-extensions install gpu

sudo mount --bind /var/lib/nvidia /var/lib/nvidia

sudo mount -o remount,exec /var/lib/nvidia

docker run \
  --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
  --volume /var/lib/nvidia/bin:/usr/local/nvidia/bin \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidiactl:/dev/nvidiactl \
-it -p 5000:5000 us.gcr.io/[YOUR_PROJECT_ID]/dl_project:v1
```
### Step five:
### Access application
Visit [EXTERNAL\_IP\_OF\_YOUR\_INSTANCE]:5000
