# PyTorch-UNet

## A much improved PyTorch-based implementation of U-Net

#### KiTS21-Pytorch.py
This class details the data structure that holds KiTS21 Challenge data and returns 2D slices of CT scans only where kidneys are present. This allows efficient indexing, facilitating batching and parallelised training of unet.

#### unet_modules.py
These classes are general forms of the blocks used to build 2D unet - down sampling convolutional (deconv), up sampling convolutional (upconv), and the output convolutional block (outconv).

#### unet.py
This class constructs a 2D unet using the generic blocks detailed in unet_modules.

#### unet_trainer.py
This script constructs an instance of unet and trains it using an instance of KiTS-Pytorch. Cost is monitored during training.

#### unet
This is an example pickle file, holding some partially-trained unet weights, that can be loaded into unet.py

#### data.csv
This is an example KiTS-Pytorch dataset, showing the schema of the pandas dataframe used to hold the filenames and slice indices that are used during training.
