#   Pixel Motion Diffusion is What We Need for Robot Control

[E-Ro Nguyen*](https://eronguyen.me) [Yichi Zhang*](https://scholar.google.com/citations?user=HOCyXzsAAAAJ&hl), [Kanchana Ranasinghe](https://kahnchana.github.io/), [Xiang Li](https://xxli.me/), [Michael S Ryoo](http://michaelryoo.com)

Stony Brook University

\*Equal contribution


  <a href="">
    <img src="https://img.shields.io/badge/CVPR-2026-blue?style=flat-square" alt="Paper">
  </a>
<a href='https://arxiv.org/abs/2509.22652'><img src='https://img.shields.io/badge/ArXiv-2509.22652-red'></a> 
<a href='https://eronguyen.github.io/DAWN/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 


<p align="center">
  <img src=".github/assets/overview.png" width="100%" alt="DAWN Overview"/>
</p>

**DAWN** a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation.

---
## 🔥 Updates
- **[May 2026]** Our code&weights has been released. 
- **[Feb 2026]** 🎉 Our paper has been accepted to **CVPR 2026**!
- **[Sep 2025]** 📄 Initial arXiv release.
---

## 🛠️ Installation
```
conda create -n dawn python=3.9 -y
conda activate dawn
pip install uv

# Install calvin as described in (https://github.com/mees/calvin). 
# Maybe you will occur some render issues and you can refer to calvin repo to solve them.
git clone --recurse-submodules https://github.com/mees/calvin.git

$ export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh

# Then install dawn requirements
cd ..
uv pip install -r requirements.txt
```

## 📷 Weights
Download our checkpoint at [here](https://huggingface.co/nero1342/DAWN/tree/main)

## Train
### Stage 1: Motion Director training
```
accelerate launch --num_processes=4 train.py config=stage1
```

### Stage 2
```
accelerate launch --num_processes=4 train.py config=stage2
```

## Inference
```
accelerate launch --num_processes=4 inference.py
```

## 📖 Citation

If you find our work useful, please consider citing:

```bibtex
@article{nguyen2025dawn,
  title   = {Pixel Motion Diffusion is What We Need for Robot Control},
  author  = {Nguyen, E-Ro and Zhang, Yichi and Ranasinghe, Kanchana and Li, Xiang and Ryoo, Michael S},
  journal = {arXiv preprint arXiv:2509.22652},
  year    = {2025}
}