import numpy as np
import os
from train_model import train_model
from prepare_dataset import prepare_dataset
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def main():

    # I/O SEARCH PARAMS
    parser = argparse.ArgumentParser(description=("arguments"))
    parser.add_argument("--quantities", nargs='+', 
                        default=["density", "temp", "vpec", "nHI"],
                        help="List of quantities saved as dictionary in dataset files")
    parser.add_argument("--output_dir", default="ml_outputs_raven", help="output folder")
    parser.add_argument("--dataset_dir", default="dataset_files")
    parser.add_argument("--dataset_file_filter", default="model_train")
    parser.add_argument("--seed", default="1234")
    
    parser.add_argument("--latent", default=True)
    parser.add_argument("--load_model", default=False)
    parser.add_argument("--image", default=False)
    parser.add_argument("--epochs", default="2")
    parser.add_argument("--num_examples", default="5000")
    parser.add_argument("--latent_dim", default="32")
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--batch_size", default="32")
    parser.add_argument('--box_sizes', action='store',
                        default=[160, 80, 40], type=int, nargs='*')

    args = parser.parse_args()
    quantities = args.quantities
    num_examples = np.int32(args.num_examples)
    epochs = np.int32(args.epochs)
    seed = np.int32(args.seed)

    batch_size = np.int32(args.batch_size)
    latent_dim = np.int32(args.latent_dim)
    lr = np.float32(args.lr)
    box_sizes = np.float32(args.box_sizes)
 
    ############################################################################
    
    dis_filters = [[64, 128, 256]]
    dis_scales = [[2,2,2]]
    
    if not args.latent:
        latent_str = ''
        gen_filters =  [[256, 128, 64], [64, 32, 32]]
        gen_scales = [[2,1,1], [2,1,1]]
    else:
        gen_filters = [[256, 256, 128], [128, 64, 64], [32, 32, 32]]
        gen_scales = [[2,2,2], [2,1,1], [2,1,1]]        
        latent_str = '_latent'+str(latent_dim)
    
    
    outputs_dir = args.output_dir+'_box'+str(len(box_sizes))+'_batch'+\
        str(batch_size)+latent_str+'/'
    
    # get dataset
    data, datasets, examples, original_dim, num_features = \
        prepare_dataset(args.dataset_dir+"/", args.dataset_file_filter, 
                        quantities, batch_size, num_examples)
    
    
    # check if the directory exists, and if not, create it
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        print(f"Directory '{outputs_dir}' created successfully.")
    else:
        print(f"Directory '{outputs_dir}' already exists.")
    
    train_model(outputs_dir, datasets, data, quantities, examples=examples, 
                        box_sizes=box_sizes, original_dim=original_dim, latent=args.latent, 
                        batch_size_per_replica=batch_size, epochs=epochs, 
                        lr=lr, latent_dim=latent_dim, 
                        dis_filters=dis_filters, dis_scales=dis_scales, 
                        gen_filters=gen_filters, gen_scales=gen_scales,
                        image=args.image, load_model=args.load_model,
                        seed=seed)
   
    ############################################################################

if __name__ == "__main__":
    main()
