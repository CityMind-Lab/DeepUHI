![DeepUHI Banner](images/banner.png "DeepUHI")

# DeepUHI: Fine-grained Urban Heat Island Effect Forecasting

This repository provides the official PyTorch implementation of our paper:

**"Fine-grained Urban Heat Island Effect Forecasting: A Context-aware Thermodynamic Modeling Framework"**

[![DOI](https://zenodo.org/badge/989924594.svg)](https://doi.org/10.5281/zenodo.15510072)

## Overview
DeepUHI is a data-driven context-aware thermodynamic modeling framework designed to forecast urban heat island (UHI) effects at a fine spatial and temporal granularity. Our approach leverages deep learning and domain knowledge to provide accurate, interpretable predictions for urban climate research and policy-making.

## Features
![DeepUHI intro.](images/DeepUHI_intro.png "DeepUHI")
- Fine-grained UHI forecasting using deep neural networks
- Context-aware thermodynamic modeling
- Decomposition of urban thermal mechanics within the modeling design
- Efficency and interpretablity

## Requirements
- Python >= 3.8
- PyTorch >= 1.4.4
- (Recommended) Linux server with CUDA 12.2 and 1x A6000 GPUs

To set up the environment:
```bash
conda create -n DeepUHI python==3.8
conda activate DeepUHI
pip install -r requirements.txt
```

## **SeoulTemp Dataset**

![SeoulTemp Intro.](images/SeoulTempDataset.png "SeoulTemp")

We collect and introduce \textit{SeoulTemp}, the first fine-grained urban temperature dataset that includes field environment data across multiple modalities. This dataset encompasses a total of 947 temperature stations covering 605 $km^{2}$ of land in Seoul from 2021 to 2024 (or later), specifically targeting spatio-temporal UHI effect forecasting at the street level in urban areas.

**Note**: SeoulTemp is continuously updated by us to include the latest urban temperature records.

To access the dataset for experiments, download it from:
- [Google Drive](https://drive.google.com/drive/folders/1IGivCmou4YJkHMkz9g_XjGCjbMkDXfx1?usp=sharing)

After downloading, place the data in the `./dataset` directory.


## Quick Start
To train and test the model, run:
```bash
python run.py
bash ./run_test.sh
```

## Web Demo
An offline demo of the SeoulUHI system is available at: [SeoUHI Web Platform](http://111.230.109.230:9222)
![demo](images/demo.gif)

## Citation
If you use this code or dataset, please cite our paper:
```
@inproceedings{DeepUHI2025,
  title={Fine-grained Urban Heat Island Effect Forecasting: A Context-aware Thermodynamic Modeling Framework},
  author={Xingchen Zou, Weilin Ruan, Siru Zhong, Yuehong Hu, & Yuxuan Liang},
  booktitle={Proceedings of the 31th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

## Acknowledgments


> We thank the Seoul City Government for the S-Dot (Seoul Data of Things) project and the provision of raw temperature data via the S-Dot and Seoul's IoT city data platform.  
>  
> The SeoulTemp dataset and SDot data platform are freely available for non-commercial academic use.  
>  
> _Disclaimer: This dataset is a personal research product and does not represent the official position of the Seoul City Government._



## Contact
For questions or collaborations, please contact: [xczou@connect.hku.hk]

