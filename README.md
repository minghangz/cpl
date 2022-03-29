# CPL: Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning

In this paper, we propose Contrastive ProposalLearning (CPL) for the weakly supervised temporal sentence grounding task. We use multiple learnable Gaussian functions to generate both positive and negative proposals within the same
video that can characterize the multiple events in a long video. Then, we propose a controllable easy to hard negative proposal mining strategy to collect negative samples within the same video, which can ease the model optimization and enables CPL to distinguish highly confusing scenes. The experiments show that our method achieves state-of-the-art performance on Charades-STA and ActivityNet Captions datasets.

Our paper was accepted by CVPR-2022. [[Paper](https://minghangz.github.io/uploads/CPL/CPL_paper.pdf)] [[Project Page](https://minghangz.github.io/publication/cpl/)]

## Pipeline

![pipeline](imgs/pipeline.png)
