## Masked Contrastive Autoencoders: A PyTorch Implementation

[comment]: <> (<p align="center">)

[comment]: <> (  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">)

[comment]: <> (</p>)

Instructions for creating conda enviroment. <br>
conda env create -f environment_mae_simclr.yml

Instructions for running code <br>
git clone https://github.com/shlokk/mae-contrastive.git <br>

python -m torch.distributed.launch main_pretrain.py --data_path /path /to /imagenet/ --output_dir mae_contrastive_baseline --log_dir mae_contrastive_baseline_logs --num_workers 8 --blr 2.5e-4 --weight_decay 0.05 --model mae_vit_base_patch16 --batch_size 64 --dist_url 'tcp://localhost:10004'

[comment]: <> (This is a PyTorch/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners]&#40;https://arxiv.org/abs/2111.06377&#41;:)

[comment]: <> (```)

[comment]: <> (@Article{MaskedAutoencoders2021,)

[comment]: <> (  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},)

[comment]: <> (  journal = {arXiv:2111.06377},)

[comment]: <> (  title   = {Masked Autoencoders Are Scalable Vision Learners},)

[comment]: <> (  year    = {2021},)

[comment]: <> (})

[comment]: <> (```)

[comment]: <> (* The original implementation was in TensorFlow+TPU. This re-implementation is in PyTorch+GPU.)

[comment]: <> (* This repo is a modification on the [DeiT repo]&#40;https://github.com/facebookresearch/deit&#41;. Installation and preparation follow that repo.)

[comment]: <> (* This repo is based on [`timm==0.3.2`]&#40;https://github.com/rwightman/pytorch-image-models&#41;, for which a [fix]&#40;https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842&#41; is needed to work with PyTorch 1.8.1+.)

[comment]: <> (### Catalog)

[comment]: <> (- [x] Visualization demo)

[comment]: <> (- [x] Pre-trained checkpoints + fine-tuning code)

[comment]: <> (- [x] Pre-training code)

[comment]: <> (### Visualization demo)

[comment]: <> (Run our interactive visualization demo using [Colab notebook]&#40;https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb&#41; &#40;no GPU needed&#41;:)

[comment]: <> (<p align="center">)

[comment]: <> (  <img src="https://user-images.githubusercontent.com/11435359/147859292-77341c70-2ed8-4703-b153-f505dcb6f2f8.png" width="600">)

[comment]: <> (</p>)

[comment]: <> (### Fine-tuning with pre-trained checkpoints)

[comment]: <> (The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU:)

[comment]: <> (<table><tbody>)

[comment]: <> (<!-- START TABLE -->)

[comment]: <> (<!-- TABLE HEADER -->)

[comment]: <> (<th valign="bottom"></th>)

[comment]: <> (<th valign="bottom">ViT-Base</th>)

[comment]: <> (<th valign="bottom">ViT-Large</th>)

[comment]: <> (<th valign="bottom">ViT-Huge</th>)

[comment]: <> (<!-- TABLE BODY -->)

[comment]: <> (<tr><td align="left">pre-trained checkpoint</td>)

[comment]: <> (<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>)

[comment]: <> (<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>)

[comment]: <> (<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth">download</a></td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">md5</td>)

[comment]: <> (<td align="center"><tt>8cad7c</tt></td>)

[comment]: <> (<td align="center"><tt>b8b06e</tt></td>)

[comment]: <> (<td align="center"><tt>9bdbb0</tt></td>)

[comment]: <> (</tr>)

[comment]: <> (</tbody></table>)

[comment]: <> (The fine-tuning instruction is in [FINETUNE.md]&#40;FINETUNE.md&#41;.)

[comment]: <> (By fine-tuning these pre-trained models, we rank #1 in these classification tasks &#40;detailed in the paper&#41;:)

[comment]: <> (<table><tbody>)

[comment]: <> (<!-- START TABLE -->)

[comment]: <> (<!-- TABLE HEADER -->)

[comment]: <> (<th valign="bottom"></th>)

[comment]: <> (<th valign="bottom">ViT-B</th>)

[comment]: <> (<th valign="bottom">ViT-L</th>)

[comment]: <> (<th valign="bottom">ViT-H</th>)

[comment]: <> (<th valign="bottom">ViT-H<sub>448</sub></th>)

[comment]: <> (<td valign="bottom" style="color:#C0C0C0">prev best</td>)

[comment]: <> (<!-- TABLE BODY -->)

[comment]: <> (<tr><td align="left">ImageNet-1K &#40;no external data&#41;</td>)

[comment]: <> (<td align="center">83.6</td>)

[comment]: <> (<td align="center">85.9</td>)

[comment]: <> (<td align="center">86.9</td>)

[comment]: <> (<td align="center"><b>87.8</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">87.1</td>)

[comment]: <> (</tr>)

[comment]: <> (<td colspan="5"><font size="1"><em>following are evaluation of the same model weights &#40;fine-tuned in original ImageNet-1K&#41;:</em></font></td>)

[comment]: <> (<tr>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">ImageNet-Corruption &#40;error rate&#41; </td>)

[comment]: <> (<td align="center">51.7</td>)

[comment]: <> (<td align="center">41.8</td>)

[comment]: <> (<td align="center"><b>33.8</b></td>)

[comment]: <> (<td align="center">36.8</td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">42.5</td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">ImageNet-Adversarial</td>)

[comment]: <> (<td align="center">35.9</td>)

[comment]: <> (<td align="center">57.1</td>)

[comment]: <> (<td align="center">68.2</td>)

[comment]: <> (<td align="center"><b>76.7</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">35.8</td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">ImageNet-Rendition</td>)

[comment]: <> (<td align="center">48.3</td>)

[comment]: <> (<td align="center">59.9</td>)

[comment]: <> (<td align="center">64.4</td>)

[comment]: <> (<td align="center"><b>66.5</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">48.7</td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">ImageNet-Sketch</td>)

[comment]: <> (<td align="center">34.5</td>)

[comment]: <> (<td align="center">45.3</td>)

[comment]: <> (<td align="center">49.6</td>)

[comment]: <> (<td align="center"><b>50.9</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">36.0</td>)

[comment]: <> (</tr>)

[comment]: <> (<td colspan="5"><font size="1"><em>following are transfer learning by fine-tuning the pre-trained MAE on the target dataset:</em></font></td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">iNaturalists 2017</td>)

[comment]: <> (<td align="center">70.5</td>)

[comment]: <> (<td align="center">75.7</td>)

[comment]: <> (<td align="center">79.3</td>)

[comment]: <> (<td align="center"><b>83.4</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">75.4</td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">iNaturalists 2018</td>)

[comment]: <> (<td align="center">75.4</td>)

[comment]: <> (<td align="center">80.1</td>)

[comment]: <> (<td align="center">83.0</td>)

[comment]: <> (<td align="center"><b>86.8</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">81.2</td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">iNaturalists 2019</td>)

[comment]: <> (<td align="center">80.5</td>)

[comment]: <> (<td align="center">83.4</td>)

[comment]: <> (<td align="center">85.7</td>)

[comment]: <> (<td align="center"><b>88.3</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">84.1</td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">Places205</td>)

[comment]: <> (<td align="center">63.9</td>)

[comment]: <> (<td align="center">65.8</td>)

[comment]: <> (<td align="center">65.9</td>)

[comment]: <> (<td align="center"><b>66.8</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">66.0</td>)

[comment]: <> (</tr>)

[comment]: <> (<tr><td align="left">Places365</td>)

[comment]: <> (<td align="center">57.9</td>)

[comment]: <> (<td align="center">59.4</td>)

[comment]: <> (<td align="center">59.8</td>)

[comment]: <> (<td align="center"><b>60.3</b></td>)

[comment]: <> (<td align="center" style="color:#C0C0C0">58.0</td>)

[comment]: <> (</tr>)

[comment]: <> (</tbody></table>)

[comment]: <> (### Pre-training)

[comment]: <> (The pre-training instruction is in [PRETRAIN.md]&#40;PRETRAIN.md&#41;.)

[comment]: <> (### License)

[comment]: <> (This project is under the CC-BY-NC 4.0 license. See [LICENSE]&#40;LICENSE&#41; for details.)
