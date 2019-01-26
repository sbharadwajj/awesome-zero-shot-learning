# Zero-Shot Learning

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources including papers, comparitive results on standard datasets and relevant links pertaining to zero-shot learning.  

## Contributing

Contributions are welcome. Please see the [issue](https://github.com/chichilicious/awesome-zero-shot-learning/issues/2) which lists the things which are planned to be included in this repo. If you wish to contribute within these boundaries, feel free to send a PR. If you have suggestions for new sections to be included, please raise an issue and discuss before sending a PR.

## Table of Contents
+ [Papers](#Papers)
+ [Datasets](#Datasets)
+ [Other Resources](#Other-resources)


## Zero-Shot Object Classification

### Papers

#### NeurIPS 2018
 + **DCN:** Shichen Liu, Mingsheng Long, Jianmin Wang, Michael I. Jordan."Generalized Zero-Shot Learning with Deep Calibration Network" NeurIPS (2018). [[pdf]](http://papers.nips.cc/paper/7471-generalized-zero-shot-learning-with-deep-calibration-network.pdf)
+ **S2GA:** Yunlong Yu, Zhong Ji, Yanwei Fu, Jichang Guo, Yanwei Pang, Zhongfei (Mark) Zhang."Stacked Semantics-Guided Attention Model for Fine-Grained Zero-Shot Learning." NeurIPS (2018). [[pdf]](http://papers.nips.cc/paper/7839-stacked-semantics-guided-attention-model-for-fine-grained-zero-shot-learning.pdf)
+ **DIPL:** An Zhao, Mingyu Ding, Jiechao Guan, Zhiwu Lu, Tao Xiang, Ji-Rong Wen "Domain-Invariant Projection Learning for Zero-Shot Recognition." NeurIPS (2018). [[pdf]](http://papers.nips.cc/paper/7380-domain-invariant-projection-learning-for-zero-shot-recognition.pdf)

#### ECCV 2018
+ **SZSL:** Jie Song, Chengchao Shen, Jie Lei, An-Xiang Zeng, Kairi Ou, Dacheng Tao, Mingli Song. "Selective Zero-Shot Classification with Augmented Attributes." ECCV (2018). [[pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jie_Song_Selective_Zero-Shot_Classification_ECCV_2018_paper.pdf)]
+ **LCP-SA:** Huajie Jiang, Ruiping Wang, Shiguang Shan, Xilin Chen. "Learning Class Prototypes via Structure Alignment for Zero-Shot Recognition." ECCV (2018). [[pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Huajie_Jiang_Learning_Class_Prototypes_ECCV_2018_paper.pdf)]
+ **MC-ZSL:** Rafael Felix, Vijay Kumar B. G., Ian Reid, Gustavo Carneiro. "Multi-modal Cycle-consistent Generalized Zero-Shot Learning." ECCV (2018). [[pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/RAFAEL_FELIX_Multi-modal_Cycle-consistent_Generalized_ECCV_2018_paper.pdf)] [[code]](https://github.com/rfelixmg/frwgan-eccv18)


#### CVPR 2018

+ **GCN:**  Xiaolong Wang, Yufei Ye, Abhinav Gupta. "Zero-shot Recognition via Semantic Embeddings and Knowledge Graphs." CVPR (2018). [[pdf](https://arxiv.org/pdf/1803.08035.pdf)] [[code](https://github.com/JudyYe/zero-shot-gcn)]
+ **PSR:** Yashas Annadani, Soma Biswas. "Preserving Semantic Relations for Zero-Shot Learning." CVPR (2018). [[pdf](https://arxiv.org/pdf/1803.03049.pdf)]
+ **GAN-NT:** Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng, Ahmed Elgammal. "A Generative Adversarial Approach for Zero-Shot Learning From Noisy Texts." CVPR (2018). [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhu_A_Generative_Adversarial_CVPR_2018_paper.pdf)]
+ **TUE:** Jie Song, Chengchao Shen, Yezhou Yang, Yang Liu, Mingli Song. "Transductive Unbiased Embedding for Zero-Shot Learning." CVPR (2018). [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Transductive_Unbiased_Embedding_CVPR_2018_paper.pdf)]
+ **SP-AEN:** Long Chen, Hanwang Zhang, Jun Xiao, Wei Liu, Shih-Fu Chang. "Zero-Shot Visual Recognition Using Semantics-Preserving Adversarial Embedding Networks." CVPR (2018). [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Zero-Shot_Visual_Recognition_CVPR_2018_paper.pdf)] [[code](https://github.com/zjuchenlong/sp-aen.cvpr18)]
+ **ML-SKG:** Chung-Wei Lee, Wei Fang, Chih-Kuan Yeh, Yu-Chiang Frank Wang. "Multi-Label Zero-Shot Learning With Structured Knowledge Graphs." CVPR (2018). [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lee_Multi-Label_Zero-Shot_Learning_CVPR_2018_paper.pdf)] [[project](https://people.csail.mit.edu/weifang/project/vll18-mlzsl/)]
+ **GZSL-SE:** Vinay Kumar Verma, Gundeep Arora, Ashish Mishra, Piyush Rai. "Generalized Zero-Shot Learning via Synthesized Examples." CVPR (2018). [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Verma_Generalized_Zero-Shot_Learning_CVPR_2018_paper.pdf)]
+ **FGN:** Yongqin Xian, Tobias Lorenz, Bernt Schiele, Zeynep Akata. "Feature Generating Networks for Zero-Shot Learning." CVPR (2018). [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_Feature_Generating_Networks_CVPR_2018_paper.pdf)] [[code](http://datasets.d2.mpi-inf.mpg.de/xian/cvpr18xian.zip)] [[project](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning/)]
+ **LDF:** Yan Li, Junge Zhang, Jianguo Zhang, Kaiqi Huang. "Discriminative Learning of Latent Features for Zero-Shot Recognition." CVPR (2018). [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Discriminative_Learning_of_CVPR_2018_paper.pdf) 
+ **WSL:** Li Niu, Ashok Veeraraghavan, and Ashu Sabharwal. "Webly Supervised Learning Meets Zero-shot Learning: A Hybrid Approach for Fine-grained Classification." CVPR (2018). [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Niu_Webly_Supervised_Learning_CVPR_2018_paper.pdf)

#### TPAMI 2018

+ **C-GUB:** Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata. "Zero-shot learning-A comprehensive evaluation of the good, the bad and the ugly." TPAMI (2018). [[pdf](https://arxiv.org/pdf/1707.00600.pdf)] [[project](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)]

#### AAAI 2018, 2017
+ **GANZrl:** Bin Tong, Martin Klinkigt, Junwen Chen, Xiankun Cui, Quan Kong, Tomokazu Murakami, Yoshiyuki Kobayashi. "Adversarial Zero-shot Learning With Semantic Augmentation." AAAI (2018). [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16805/15965)
+ **JDZsL:** Soheil Kolouri, Mohammad Rostami, Yuri Owechko, Kyungnam Kim. "Joint Dictionaries for Zero-Shot Learning." AAAI (2018). [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16404/16723)
+ **VZSL:** Wenlin Wang, Yunchen Pu, Vinay Kumar Verma, Kai Fan, Yizhe Zhang, Changyou Chen, Piyush Rai, Lawrence Carin. "Zero-Shot Learning via Class-Conditioned Deep Generative Models." AAAI (2018). [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16087/16709)
+ **AS:** Yuchen Guo, Guiguang Ding, Jungong Han, Sheng Tang. "Zero-Shot Learning With Attribute Selection." AAAI (2018). [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16350/16272)
+ **DSSC:** Yan Li, Zhen Jia, Junge Zhang, Kaiqi Huang, Tieniu Tan."Deep Semantic Structural Constraints for Zero-Shot Learning." AAAI (2018). [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16309/16294)
+ **ZsRDA:** Yang Long, Li Liu, Yuming Shen, Ling Shao. "Towards Affordable Semantic Searching: Zero-Shot Retrieval via Dominant Attributes." AAAI (2018). [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16626/16314)
+ **DCL:** "Zero-Shot Recognition via Direct Classifier Learning with Transferred Samples and Pseudo Labels." AAAI (2017). [[pdf]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14160/14281)


#### ICCV 2017
+ **A2C:** Berkan Demirel, Ramazan Gokberk Cinbis, Nazli Ikizler-Cinbis. "Attributes2Classname: A Discriminative Model for Attribute-Based Unsupervised Zero-Shot Learning." ICCV (2017). [[pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Demirel_Attributes2Classname_A_Discriminative_ICCV_2017_paper.pdf)] [[code]](https://github.com/berkandemirel/attributes2classname)
+ **PVE:** Soravit Changpinyo, Wei-Lun Chao, Fei Sha. "Predicting Visual Exemplars of Unseen Classes for Zero-Shot Learning." ICCV (2017). [[pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Changpinyo_Predicting_Visual_Exemplars_ICCV_2017_paper.pdf)][[code](https://github.com/pujols/Zero-shot-learning-journal)]
+ **LDL:** Huajie Jiang, Ruiping Wang, Shiguang Shan, Yi Yang, Xilin Chen. "Learning Discriminative Latent Attributes for Zero-Shot Classification." ICCV (2017). [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Jiang_Learning_Discriminative_Latent_ICCV_2017_paper.pdf)]

#### CVPR 2017

+ **GUB:** Yongqin Xian, Bernt Schiele, Zeynep Akata. "Zero-Shot learning - The Good, the Bad and the Ugly." CVPR (2017). [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xian_Zero-Shot_Learning_-_CVPR_2017_paper.pdf)] [[code](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip)]
+ **Deep-SCoRe:** Pedro Morgado, Nuno Vasconcelos."Semantically Consistent Regularization for Zero-Shot Recognition." CVPR (2017). [[pdf](http://www.svcl.ucsd.edu/~morgado/score/score-cvpr17.pdf)] [[code](https://github.com/pedro-morgado/score-zeroshot)]
+ **DEM:** Li Zhang, Tao Xiang, Shaogang Gong. "Learning a Deep Embedding Model for Zero-Shot Learning." CVPR (2017). [[pdf](https://arxiv.org/pdf/1611.05088.pdf)] [[code](https://github.com/lzrobots/DeepEmbeddingModel_ZSL)]
+ **VDS:** Yang Long, Li Liu, Ling Shao, Fumin Shen, Guiguang Ding, Jungong Han. "From Zero-Shot Learning to Conventional Supervised Classification: Unseen Visual Data Synthesis." CVPR (2017). [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/html/Long_From_Zero-Shot_Learning_CVPR_2017_paper.html)]
+ **ESD:** Zhengming Ding, Ming Shao, Yun Fu. "Low-Rank Embedded Ensemble Semantic Dictionary for Zero-Shot Learning." CVPR (2017). [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ding_Low-Rank_Embedded_Ensemble_CVPR_2017_paper.pdf)]
+ **SAE:** Elyor Kodirov, Tao Xiang, Shaogang Gong. "Semantic Autoencoder for Zero-Shot Learning." CVPR (2017). [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kodirov_Semantic_Autoencoder_for_CVPR_2017_paper.pdf) [[code](https://github.com/Elyorcv/SAE)]
+ **DVSM:** Yanan Li, Donghui Wang, Huanhang Hu, Yuetan Lin, Yueting Zhuang. "Zero-Shot Recognition Using Dual Visual-Semantic Mapping Paths". CVPR (2017). [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Zero-Shot_Recognition_Using_CVPR_2017_paper.pdf)]
+ **MTF-MR:** Xing Xu, Fumin Shen, Yang Yang, Dongxiang Zhang, Heng Tao Shen, Jingkuan Song. "Matrix Tri-Factorization With Manifold Regularizations for Zero-Shot Learning." CVPR (2017). [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Matrix_Tri-Factorization_With_CVPR_2017_paper.pdf)]
+ Nour Karessli, Zeynep Akata, Bernt Schiele, Andreas Bulling. "Gaze Embeddings for Zero-Shot Image Classification." CVPR (2017). [[pdf]](https://arxiv.org/pdf/1611.09309.pdf) [[code]](https://github.com/Noura-kr/CVPR17)

#### CVPR 2016

+ **MC-ZSL:** Zeynep Akata, Mateusz Malinowski, Mario Fritz, Bernt Schiele. "Multi-Cue Zero-Shot Learning With Strong Supervision." CVPR (2016). [[pdf] (http://openaccess.thecvf.com/content_cvpr_2016/papers/Akata_Multi-Cue_Zero-Shot_Learning_CVPR_2016_paper.pdf)] [[code]](https://www.mpi-inf.mpg.de/index.php?id=2935)
+ **LATEM:** Yongqin Xian, Zeynep Akata, Gaurav Sharma, Quynh Nguyen, Matthias Hein, Bernt Schiele. "Latent Embeddings for Zero-Shot Classification." CVPR (2016). [[pdf](http://openaccess.thecvf.com/content_cvpr_2016/papers/Xian_Latent_Embeddings_for_CVPR_2016_paper.pdf)][[code](http://datasets.d2.mpi-inf.mpg.de/yxian16cvpr/latEm.zip)]
+ **LIM:** Ruizhi Qiao, Lingqiao Liu, Chunhua Shen, Anton van den Hengel. "Less Is More: Zero-Shot Learning From Online Textual Documents With Noise Suppression." CVPR (2016). [[pdf](http://openaccess.thecvf.com/content_cvpr_2016/papers/Qiao_Less_Is_More_CVPR_2016_paper.pdf)]
+ **SYNC:** Soravit Changpinyo, Wei-Lun Chao, Boqing Gong, Fei Sha. "Synthesized Classifiers for Zero-Shot Learning." CVPR (2016). [[pdf](http://openaccess.thecvf.com/content_cvpr_2016/papers/Changpinyo_Synthesized_Classifiers_for_CVPR_2016_paper.pdf)][[code](https://github.com/pujols/zero-shot-learning)]
+ **RML:** Ziad Al-Halah, Makarand Tapaswi, Rainer Stiefelhagen. "Recovering the Missing Link: Predicting Class-Attribute Associations for Unsupervised Zero-Shot Learning." CVPR (2016). [[pdf](http://openaccess.thecvf.com/content_cvpr_2016/papers/Al-Halah_Recovering_the_Missing_CVPR_2016_paper.pdf)]
+ **SLE:** Ziming Zhang, Venkatesh Saligrama. "Zero-Shot Learning via Joint Latent Similarity Embedding." CVPR (2016). [[pdf](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Zero-Shot_Learning_via_CVPR_2016_paper.pdf)] [[code](https://drive.google.com/file/d/1RimUgUlf2tfpntzlxdlYaAvm34HX0fUb/view?usp=sharing)]


#### ECCV 2016
+ Wei-Lun Chao, Soravit Changpinyo, Boqing Gong2, Fei Sha. "An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild." ECCV (2016). [[pdf]](https://arxiv.org/pdf/1605.04253.pdf)
+ **MTE:** Xun Xu, Timothy M. Hospedales, Shaogang Gong. "Multi-Task Zero-Shot Action Recognition with Prioritised Data Augmentation." ECCV (2016). [[pdf]](https://arxiv.org/pdf/1611.08663.pdf)
+ Ziming Zhang, Venkatesh Saligrama."Zero-Shot Recognition via Structured Prediction." ECCV (2016). [[pdf]](https://pdfs.semanticscholar.org/be96/1637db8561b027fd48788c64e7919c7cd760.pdf)
+ Maxime Bucher, Stephane Herbin, Frederic Jurie."Improving Semantic Embedding Consistency by Metric Learning for Zero-Shot Classification." ECCV (2016). [[pdf]](https://arxiv.org/pdf/1607.08085.pdf)

#### AAAI 2016
+ **RKT:** Donghui Wang, Yanan Li, Yuetan Lin, Yueting Zhuang. "Relational Knowledge Transfer for Zero-Shot Learning." AAAI (2016). [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11802/11854)


#### CVPR 2015
+ Zeynep Akata, Scott Reed, Daniel Walter, Honglak Lee, Bernt Schiele. "Evaluation of Output Embeddings for Fine-Grained Image Classification." CVPR (2015). [[pdf]](https://www.mpi-inf.mpg.de/fileadmin/inf/d2/akata/1690.pdf) [[code]](https://www.mpi-inf.mpg.de/index.php?id=2325) 
+ Zhenyong Fu, Tao Xiang, Elyor Kodirov, Shaogang Gong. "Zero-Shot Object Recognition by Semantic Manifold Distance." CVPR (2015). [[pdf]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Fu_Zero-Shot_Object_Recognition_2015_CVPR_paper.pdf)

#### ICCV 2015
+ **SSE:** Ziming Zhang, Venkatesh Saligrama. "Zero-Shot Learning via Semantic Similarity Embedding." ICCV (2015). [[pdf]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhang_Zero-Shot_Learning_via_ICCV_2015_paper.pdf) [[code]](https://zimingzhang.wordpress.com/source-code/)
+ **LRL:** Xin Li, Yuhong Guo, Dale Schuurmans."Semi-Supervised Zero-Shot Classification with Label Representation Learning." ICCV (2015). [[pdf]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Li_Semi-Supervised_Zero-Shot_Classification_ICCV_2015_paper.pdf)
+ **UDA:** Elyor Kodirov, Tao Xiang, Zhenyong Fu, Shaogang Gong. "Unsupervised Domain Adaptation for Zero-Shot Learning." ICCV (2015). [[pdf]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kodirov_Unsupervised_Domain_Adaptation_ICCV_2015_paper.pdf)
+ Jimmy Lei Ba, Kevin Swersky, Sanja Fidler, Ruslan Salakhutdinov. "Predicting Deep Zero-Shot Convolutional Neural Networks using Textual Descriptions." ICCV (2015). [[pdf]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ba_Predicting_Deep_Zero-Shot_ICCV_2015_paper.pdf)

#### TPAMI 2015
+ **TMV:** Yanwei Fu, Timothy M. Hospedales, Tao Xiang, Shaogang Gong. "Transductive Multi-view Zero-Shot Learning." TPAMI (2015) [[pdf]](https://arxiv.org/pdf/1501.04560.pdf) [[code]](https://github.com/yanweifu/embedding_zero-shot-learning)

#### NIPS 2014, 2013, 2009
+ Dinesh Jayram, Kristen Grauman."Zero-Shot Recognition with Unreliable Attributes" NIPS (2014) [[pdf]](https://papers.nips.cc/paper/5290-zero-shot-recognition-with-unreliable-attributes.pdf)
+ **CMT:** Richard Socher, Milind Ganjoo, Christopher D. Manning, Andrew Y. Ng. "Zero-Shot Learning Through Cross-Modal Transfer" NIPS (2013) [[pdf]](https://papers.nips.cc/paper/5027-zero-shot-learning-through-cross-modal-transfer.pdf)  [[code]](https://github.com/mganjoo/zslearning)
+ **DeViSE:** Andrea Frome, Greg S. Corrado, Jonathon Shlens, Samy Bengio, Jeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov."DeViSE: A Deep Visual-Semantic Embedding Model" NIPS (2013) [[pdf]](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)
+ Mark Palatucci, Dean Pomerleau, Geoffrey Hinton, Tom M. Mitchell. "Zero-Shot Learning with Semantic Output Codes" NIPS (2009) [[pdf]](https://papers.nips.cc/paper/3650-zero-shot-learning-with-semantic-output-codes.pdf)

#### ECCV 2014
+ **TMV-BLP:** Yanwei Fu, Timothy M. Hospedales, Tao Xiang, Zhenyong Fu, Shaogang Gong. "Transductive Multi-view Embedding for
Zero-Shot Recognition and Annotation" ECCV (2014).[[pdf]](https://www.eecs.qmul.ac.uk/~txiang/publications/Fu_et_al_embedding_eccv2014.pdf) [[code]](https://github.com/yanweifu/embedding_zero-shot-learning)
+ Stanislaw Antol, Larry Zitnick, Devi Parikh. "Zero-Shot Learning via Visual Abstraction." ECCV (2014). [[pdf]](https://computing.ece.vt.edu/~santol/projects/zsl_via_visual_abstraction/eccv2014_zsl_via_visual_abstraction.pdf) [[code]](https://github.com/StanislawAntol/zsl_via_visual_abstraction) [[project]](https://computing.ece.vt.edu/~santol/projects/zsl_via_visual_abstraction/)

#### Other Papers
+ **EsZSL:** Bernardino Romera-Paredes, Philip H. S. Torr. "An embarrassingly simple approach to zero-shot learning." ICML (2015). [[pdf]](http://proceedings.mlr.press/v37/romera-paredes15.pdf) [[Code]](https://github.com/MLWave/extremely-simple-one-shot-learning)
+ **AEZSL:** "Zero-Shot Learning via Category-Specific Visual-Semantic Mapping and Label Refinement" IEEE SPS (2018). [[pdf]](https://ieeexplore.ieee.org/document/8476580)
+ **ZSGD:** Tiancheng Zhao, Maxine Eskenazi. "Zero-Shot Dialog Generation with Cross-Domain Latent Actions" SIGDIAL (2018). [[pdf]](https://arxiv.org/abs/1805.04803v1) [[code]](https://github.com/snakeztc/NeuralDialog-ZSDG)
+ Yanwei Fu, Tao Xiang, Yu-Gang Jiang, Xiangyang Xue, Leonid Sigal, Shaogang Gong "Recent Advances in Zero-shot Recognition".  IEEE Signal Processing Magazine. [[pdf]](https://arxiv.org/pdf/1710.04837.pdf)
+ Michael Kampffmeyer, Yinbo Chen, Xiaodan Liang, Hao Wang, Yujia Zhang, Eric P. Xing "Rethinking Knowledge Graph Propagation for Zero-Shot Learning" arXiv (2018). [[pdf]](https://arxiv.org/pdf/1805.11724v2.pdf) [[code]](https://github.com/cyvius96/adgpm)

### Datasets
+ **LAD:** Large-scale Attribute Dataset. Categories:230. [[link]](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset-for-Zero-shot-Learning)
+ **CUB:** Caltech-UCSD Birds. Categories:200. [[link]](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
+ **AWA2:** Animals with Attributes. Categories:50. [[link]](https://cvml.ist.ac.at/AwA2/)
+ **aPY:** attributes Pascal and Yahoo. Categories:32 [[link]](http://vision.cs.uiuc.edu/attributes/)
+ **Flowers Dataset:** There are two datasets, Categories: 17 and 102. [[link]](http://www.robots.ox.ac.uk/~vgg/data/flowers/)
+ **SUN:** Scene Attributes. Categories:717. [[link]](http://cs.brown.edu/~gmpatter/sunattributes.html)



## Other resources
+ https://medium.com/@alitech_2017/from-zero-to-hero-shaking-up-the-field-of-zero-shot-learning-c43208f71332
+ https://www.analyticsindiamag.com/what-is-zero-shot-learning/
+ https://medium.com/@cetinsamet/zero-shot-learning-53080995d45f

## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
