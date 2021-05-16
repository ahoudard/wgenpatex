import argparse
import wgenpatex
import model
import torch

parser = argparse.ArgumentParser()
parser.add_argument('target_image_path', help='paths of target texture image')
parser.add_argument('-w', '--patch_size', type=int,default=4, help="patch size (default: 4)")
parser.add_argument('-nmax', '--n_iter_max', type=int, default=5000, help="max iterations of the algorithm(default: 5000)")
parser.add_argument('-npsi', '--n_iter_psi', type=int, default=10, help="max iterations for psi (default: 10)")
parser.add_argument('-nin', '--n_patches_in', type=int, default=-1, help="number of patches of the synthetized texture used at each iteration, -1 corresponds to all patches (default: -1)")
parser.add_argument('-nout', '--n_patches_out', type=int, default=2000, help="number maximum of patches of the target texture used, -1 corresponds to all patches (default: 2000)")
parser.add_argument('-sc', '--scales', type=int, default=5, help="number of scales used (default: 5)")
parser.add_argument('--visu',  action='store_true', help='show intermediate results')
parser.add_argument('--save',  action='store_true', help='save temp results in /tmp folder')
parser.add_argument('--keops', action='store_true', help='use keops package')
args = parser.parse_args()

generator = wgenpatex.learn_model(args)

# save the texture generator
torch.save(generator.state_dict(), 'generator.pt')

# sample an image and save it
synth_img = model.sample_fake_img(generator, [1,3,512,512] , n_samples=1)
wgenpatex.imshow(synth_img)
wgenpatex.imsave('synthesized.png', synth_img)
