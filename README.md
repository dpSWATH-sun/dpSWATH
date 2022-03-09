
# dpSWATH
dpSWATH is developed for building high quality library for SWATH-MS based on deep-learning.
<div align=center><img src='/pics/dpSWATH_start.png'/></div>


## Major functions
* Preprocess data of high-quality identified fragmentations, retention time.
* Train models for predictions of retention time and mass spectra combined with dpMC for local specific experiments with fine-tuning.
* Building high quality library for SWATH-MS analysis using trained model.


## System requirements & Runtime Environment(RE) Confirmation
* Both of Windows and Linux platforme are supported.
* NVIDIA Graphics Processing Unit (GPU) is highly reconmmended; Central Processing Unit (CPU) calculation is also available but depreciated;
* NVIDIA CUDA 10.0+, CuDNN 7.6.5+ are recomended.
* Keras with Tensorflow backend.

dpSWATH was developed under Python 3.6.5(Anaconda3 5.2.0 64-bit) with keras tensorflow-gpu backend. Hardware including GPU card NVIDIA GeForce 1080Ti, CPU i7-8086K and 128GB RAM were utilized. 


## Installation 
**1. Installation of Python (Anaconda is recommended)**

   * Anaconda installer can be downloaded from the [Anaconda official site](https://www.anaconda.com/products/individual).
   * Official python installer can be downloaded from the [Python official site](https://www.python.org/downloads/).

**2. Installation of associated packages**

   * Install Tensorflow using `conda install(recommended)` or pip:
   
      * *`conda install -c conda-forge tensorflow`* or *pip install --upgrade tensorflow*
   
   * Install Tensorflow with GPU supported using `conda install(recommended)` or pip:
   
      * *`conda install -c anaconda tensorflow-gpu`* or *pip install --upgrade tensorflow-gpu*
 
   * Install Keras using `conda install(recommended)` or pip:
   
      * *`conda install -c conda-forge keras`* or *pip install keras*

   * Other associated packages including **``os,re,datetime,Bio,pandas,numpy,random,fnmatch``** can also be installed using `conda install(recommended)` or pip.


## Files needed for dpRT
* For the training of dpRT model, **retention time files** from the following searching software are supported:
   * **SpectroMine/Spectronaut**, `the experimental library(.xls) file built by Pulsar in Spectronaut or searching file from SpectroMine` are supported.
   * **ProteinPilot**, `the identifications from ".mzid" file generated from ProteinPilot` is supported. 
* **Pretrained model** for fine-tuning. Fine-tuning is provided when training model of dpRT and the pretrained models are provided in the [models](models/) folder.
* **Trained model** for prediction of retention time. This file is needed when you have trained your dpRT model and ready to build dpSWATH library.

## Files needed for dpMS
* For the training of dpMS model, **mass spectra files** from the following searching software are supported:
   * **SpectroMine/Spectronaut**, `the experimental library(.xls) file built by Pulsar in Spectronaut or searching file from SpectroMine` are supported.
   * **ProteinPilot**, `the identifications from ".mzid" file generated from ProteinPilot` is supported.
* **Pretrained model** for fine-tuning. Fine-tuning is provided when training model of dpMS and the pretrained models are provided in the [models](models/) folder.
* **Trained model** for prediction of mass spectra. This file is needed when you have trained your dpMS model and ready to build dpSWATH library.


## Preprocessing of datasets
For the training of both dpRT and dpMS, only high quality data are used to train the models. 
* For the datasets used for dpRT, only the retention time of high confident peptides are selected.
* For the datasets used for dpMS, dpMScore are performed to get the consistent mass spctra for the training of dpMS.


## Procedures to train models of dpRT
1) Start using dpSWATH by opening command interpreter *`cmd.exe`* in windows platform or *`shell`* in Liux platform.
2) Run dpSWATH by calling *`python`* program: `python dpSWATH_main.py`.
3) After entering the commond line, follow the prompt and enter `1` to select `training` models.
4) Then select `1` to train dpRT.
5) Next, please set your working directory after the prompt which will store all your trained dpRT models.
6) Put the absolute path of your `pretrained dpRT model` and  `DDA library` after corresponding prompt.
7) The trained dpRT models can be found under folder `./working directory/dpSWATH/md/dpRT/XXX-XX-XX_XX_XX_XX_XXXXXX/`, please keep the best model based on the 'validation loss' for building library.

## Procedures to train models of dpMS
1) Start using dpSWATH by opening command interpreter *`cmd.exe`* in windows platform or *`shell`* in Liux platform.
2) Run dpSWATH by calling *`python`* program: `python dpSWATH_main.py`.
3) After entering the commond line, follow the prompt and enter `1` to select `training` models.
4) Then select `2` to train dpMS.
5) Next, please set your working directory after the prompt which will store all your trained dpRT models.
6) Put the absolute path of your `pretrained dpMS model` and  `DDA fragmatation file from ProteinPilot` after corresponding prompt.
7) The trained dpRT models can be found under folder `./working directory/dpSWATH/md/dpMS/XXX-XX-XX_XX_XX_XX_XXXXXX/`, please keep the best model based on the 'validation loss' for building library.

  * The examples for training of dpRT model:
<div align=center><img src='/pics/train_dpRT.PNG'/></div>


  * The examples for training of dpMS model:
<div align=center><img src='/pics/train_dpMS.PNG'/></div>


## Procedures to build dpSWATH library
1) Start using dpSWATH by opening command interpreter *`cmd.exe`* in windows platform or *`shell`* in Liux platform.
2) Run dpSWATH by calling *`python`* program: `python dpSWATH_main.py`.
3) After entering the commond line, follow the prompt and enter `2` to select `build library`.
4) Next, please set your working directory after the prompt which will store the built dpSWATH library.
5) Put the absolute path of your `file of precursors` and directory of `models of dpRT and dpMS` after corresponding prompt.
6) The result can be found under folder `./working directory/dpSWATH/Library/dpSWATH-Lib.txt`.

  **The progression informaiton will be shown in the progress bar in commond window.**

* The examples for building library by dpSWATH:
<div align=center><img src='/pics/bld_lib.PNG'/></div>


## Notes for the files generated by dpSWATH

* **The model files**. The model files can be generated during training process by selecting *`train`* function. All the model files are in the following format:
`./dpRT_XXX_Y.YYYYY_Z.ZZZZZ.h5/` or `./dpMS_XXX_Y.YYYYY_Z.ZZZZZ.h5/`, in which `XXX` denotes the *epoch* of the model, `Y.YYYYY` denotes the *training loss* for this epoch of training, `Z.ZZZZZ` denotes the *validation loss* for this epoch of training. 
   * For the `training` function, the model files can be found under directory `./working directory/dpSWATH/md/dpRT/XXX-XX-XX_XX_XX_XX_XXXXXX/` or `./working directory/dpSWATH/md/dpMS/XXX-XX-XX_XX_XX_XX_XXXXXX/`.

* **The precursor file**. The precursor file used for building dpSWATH library has two columns: 1) the peptides; 2) the precursor charge. We recommend to prepare the precursor lists by using dpMC (https://github.com/dpMC-sun/dpMC).

* **The library files**. Please refer to the Supplementary Note of our paper.
   * For the `build library` function, the digested file can be found in `./working directory/dpSWATH/Library/dpSWATH-Lib.txt`.


## Contacts
Please submit any issues happened to dpMC on issues section or send Emails to dpSWATH.sun@gmail.com.
