# Cars Generator
**DCGAN, WHAN-GP, RAGAN, SAGAN, StyleGAN, ...**

(polishing & uploading old project + learning a bunch of new things about GANs)

cars come in multiple colors, shapes and attitutes
we have white, black (zoomies included) and god forbid oranges
some combinations include cows, tuxedos and 
the sure thing is they make our FYPs much more intertaining
the aim of this project is to create more of'em 

# Cool findings
<table>
  <tr>
    <td align="center">
      <img src="images/ragan_mid_training_interpolation.png" width="80%"><br>
      interpolation upsampling RAGAN Generated images mid training<br>  
    </td>
    <td align="center">
      <img src="images/ragan_mid_training_deconv.png" width="80%"><br>
      DeConv upsampling RAGAN Generated images mid training<br>  
    </td>
  </tr>
</table>



# Samples
**(exploiting kaggle until someone gift me a cluster to practive DDP and spend a couple of months building an on the fly optimization framework for parallel experimentations)**
### Base Model
(demons that look like cats when u squint your eyes)
<p align="center">
  <img src="images/GANs-samples.png" width="720"><br>
  <em>selected samples from different models and configs at varying training phases</em>
</p>

## Optimization
#### Data pipeline: Loading + Augmentation 3x speedup (minima optimization for both)

in the `data-optimization branch` I switched from the conventional `open-cv`, `albumentation` and `pytorch`'s DataLoaders
to `Nvidia-dali`: data loading (nvidia never failed to amaze me) and `Kornia` augmentation on device a tresure I found

#### concurrent computation of losses and penalties (when possible :'))
*(experiemnts with cuda streams wheren't robust (depending on torch.compile) thus only used in optimization experiementations :'( )*



<br>
<p align="center">
  <img src="images/rick_n_morty.jpg" width="640"><br>
  <em>admire the stars morty (none atm)</em>
</p>
