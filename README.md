# StructFormer: Structure-Consistent Face De-Identification under Strong Privacy Constraints
 
Official PyTorch implementation of [StructFormer: Structure-Consistent Face De-Identification under Strong Privacy Constraints](https://openaccess.thecvf.com/content/WACV2026W/GAPBio/papers/Zhu_StructFormer_Structure-Consistent_Face_De-Identification_under_Strong_Privacy_Constraints_WACVW_2026_paper.pdf)

## Installation

Please download the code:

To use our code, first download the repository:
````
git clone https://github.com/zhn-nicole/Structormer.git
````

To install the dependencies:

````
pip install -r requirements.txt
````
````
Before training, please download the weights. Baidu Netdisk link: 
链接: https://pan.baidu.com/s/1uUa8SEvDdtENTV53gZzUwQ?pwd=4wfm 提取码: 4wfm
The directories are respectively:
source/shape_predictor_68_face_landmarks.dat
facenet_pytorch/20180402-114759-vggface2.pt
facenet_pytorch/20180408-102900-casia-webface.pt
dataset/celeba/modelG_ciagan.pth
````

## Training

In order to train a Structormer model, run the following command:

````
python train.py
````

We provided an example of our dataset that contains 5 identity folders from celebA dataset in the dataset folder. To train with full celebA dataset (or your own dataset), please setup the data in the same format. For the results generated in our paper, we trained the network using 1200 identities (each of them having at least 30 images) from celebA dataset. The identities can be found in: 



````
dataset/celeba/legit_indices.npy
````


We provide example of inference code in test.py file:

````
python test.py --model [path to the model and its name] --data [path to the data (optional)] -out [path to the output directory (optional)]
````


To process landmarks you can use code in process_data.py:
````
python process_data.py --input [path to a directory with raw data] --output [path to the output directory] -dlib [path to the dlib shape detector model(optional)]
````



## Citation

If you find this code useful, please consider citing the following paper:

````
@inproceedings{zhu2026structformer,
  title={StructFormer: Structure-Consistent Face De-Identification under Strong Privacy Constraints},
  author={Zhu, Haini and Jain, Deepak Kumar and Zhao, Xudong and Li, Muyu and Struc, Vitomir and Tyagi, Sumarga Kumar Sah},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1663--1673},
  year={2026}
}

````
