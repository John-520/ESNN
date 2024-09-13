# ESNN
Models for ESNN
The code can be found in the master branch


# [AEI 2024] ESNN code

This is the source code for "<b>Envelope spectrum neural network with adaptive domain weight harmonization for intelligent bearing fault diagnosis under cross-machine scenarios</b>". 

## Abstract
Accurate bearing fault diagnosis technology is highly important for ensuring the safe operation of mechanical equipment. Fault diagnosis methods can be roughly divided into signal processing-based methods (SPM) and data-driven methods (DDMs), which rely on physical knowledge and data knowledge, respectively. However, SPM cannot adapt to large data samples and dynamic parameter variations. DDMs did not consider physical representation consistency. To address the above issues, an envelope spectrum neural network (ESNN) is proposed for cross-machine bearing fault diagnosis. First, a deep transfer convolution network (DTCN) is constructed as the basic diagnostic framework. Second, a knowledge-to-model conversion strategy (KMC) is built to create ESNN, which utilizes equivalent convolution functions to construct the first two layers of DTCN, guiding the model to learn physical knowledge. Subsequently, an adaptive domain weight harmonization (ADWH) mechanism is proposed that can dynamically combine marginal distributions and joint distributions and automatically learn hidden features contained in physical and data knowledge, thus alleviating domain shift issues. Experimental evaluations are conducted using fault datasets from three different machines. The results show that, compared to models without knowledge guidance, ESNN achieves a 6.4% improvement in diagnostic accuracy. Compared to many advanced models, ESNN can achieve a maximum diagnostic accuracy improvement of 35.94%.

## Proposed Network

![image](https://github.com/user-attachments/assets/7827a14d-8deb-4fdf-95be-b94e9642cf8a)





## Dataset Preparation

**You can find the dataset here:
【1】Case Western Reserve University Bearing Data Center Website [Online] Available: http://csegroups.case.edu/bearingdatacenter/home [DB]. 
【2】Intelligent Maintenance System Bearing Dataset [Online] Available: https://www.nasa.gov/intelligent-systems-division/ [DB].**


### cross-machine

For example:

```python
mian.py
---dataloaders  #dataset
'输入数据'
# S_train_data 源域训练数据
# T_train_data 目标域训练数据
# T_test_data 目标域测试数据
```

## Contact

If you have any questions, please feel free to contact me:

- **Name:** Feiyu Lu
- **Email:** 21117039@bjtu.edu.cn
- **微信公众号:** 轴承智能故障诊断<img width="300" alt="二维码" src="https://github.com/user-attachments/assets/77a67e89-3214-4ff4-8256-01c75ec49e4b">


## Citation

If you find this paper and repository useful, please cite our paper 😊.

```
@article{LU2024102787,
title = {Envelope spectrum neural network with adaptive domain weight harmonization for intelligent bearing fault diagnosis under cross-machine scenarios},
journal = {Advanced Engineering Informatics},
volume = {62},
pages = {102787},
year = {2024},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2024.102787},
url = {https://www.sciencedirect.com/science/article/pii/S147403462400435X},
author = {Feiyu Lu and Qingbin Tong and Xuedong Jiang and Shouxin Du and Jianjun Xu and Jingyi Huo and Ziheng Zhang},
keywords = {Fault diagnosis, Envelope spectrum, Domain adaptation, Cross-machine},
}

```

```
@article{LU2024102536,
title = {Towards multi-scene learning: A novel cross-domain adaptation model based on sparse filter for traction motor bearing fault diagnosis in high-speed EMU},
journal = {Advanced Engineering Informatics},
volume = {60},
pages = {102536},
year = {2024},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2024.102536},
url = {https://www.sciencedirect.com/science/article/pii/S1474034624001848},
author = {Feiyu Lu and Qingbin Tong and Jianjun Xu and Ziwei Feng and Xin Wang and Jingyi Huo and Qingzhu Wan},
keywords = {Bearing fault diagnosis, Sparse filter, Cross-domain adaptation},
```
