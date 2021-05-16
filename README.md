# wgenpatex
Wasserstein Generative Models for Patch-based Texture Synthesis

Code related to the paper https://arxiv.org/abs/2007.03408

- wgenpatex.py contains all the functions
- model.py contains the generative model architecture
- syntax `python run_optim_synthesis.py target_image_path --options` for running texture synthesis with image optimisation (Alg.1 from paper)
- syntax `python run_cnn_synthesis.py target_image_path --options` for learning an convolutional neural network texture generator (Alg.2 from paper) (GPU recommanded)

replace --options with any:

-w or --patch_size *patch size*
-nmax or --n_iter_max *max iterations of the algorithm*
-npsi or --n_iter_psi *max iterations for psi*
-nin or --n_patches_in *number of patches of the synthetized texture used at each iteration, -1 corresponds to all patches*
-nout or --n_patches_out *number of patches of the target texture used at each iteration, -1 corresponds to all patches*
-sc or --scales *number of scales used*

--visu *plot intermediate results*
--save *save intermediate results in /tmp folder*
--keops' *use keops package (speed-up)*
