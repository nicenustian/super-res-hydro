# Super Resolution one dimnesional fields using Multiscale Wasserstein Generative Adversarial Network with Gradient Penalty (MS-WGAN-GP)

Generate super resolution one dimensional fields using Multiscale Wasserstein Generative Adversarial Network with Gradient Penalty (MS-WGAN-GP) with converging statistical properties such as power spectrum and PDF from outputs of Hydrodynamical simulations 

## Problem Statement

The Hydrodynamical Simulations are limited in terms of number of particles/cells can used to represent a part of Universe. One way is to simulate different volumes of Universe with same number of particles. Later, using MS-WGAN-GP to paint the properties of smaller volumes onto larger all with same cells/particles.  However, we need to have these fields with similar statistical properties at different volumnes. Therefore, we empoly statistical metrics such as power spectrum and PDF to converge at different resolutions. Below we have shown the convergence of these metrics and the qunatities during training (1000 epochs). The training data is from simulations with Volumes 160Mpc/h, 80Mpc/h and 40Mpc/h all with same particles of 2048^3.  

## Lessons Learned

1. Keep learning rate around 1e-4 and epochs more than 500 for higher resolution (for exmaple 2048 pixels) 1 dimensional spectra for better convergence.
2. Adam optimized does a good job but RMSprop is good alternative as well and less prone to outliers.
3. Add power spectrum and PDF losses to both Discriminator and Generator.
4. Initially the learning is limited by adverserial losses but in later stages by PDF and Power spectrum losses contribution, which is very important to generate one dimensional data with the statistical properties. 
5. There is no need to train Disciminator for more epochs, you can train Discrimnator and Generator together.

``` command
python main.py --epochs 1000 --lr 1e-4 --output_dir ml_output --dataset_dir dataset 
```

## Power spectrum of one dimensional fields
https://github.com/nicenustian/super-res-hydro/assets/111900566/0e0f5a70-54c6-4cd7-960a-3aa2aeb4e3b8

## PDF of one dimensional fields
https://github.com/nicenustian/super-res-hydro/assets/111900566/99e0421a-6212-4322-9fa6-88ad47e25f95

## velocities
https://github.com/nicenustian/super-res-hydro/assets/111900566/6a95ffc3-c257-4459-b04c-01c796bd8a7b

## gas density field
https://github.com/nicenustian/super-res-hydro/assets/111900566/18fb8fa5-ca0f-4fee-a5e9-2d863fccef7c

## temp field
https://github.com/nicenustian/super-res-hydro/assets/111900566/c41c63de-7411-4060-853a-8a00f4b50839




Please provide the one dimensional fields for each simulation in dataset_dir folder. Each file should be either .npy or .hdf5. The field name's should be output as dictionary, see example below. Each file is simulation ran at particular volume. Please make sure the code reads them in order of lowest to highest resolution. The order of fields read is displayed in the start. If it is not read in order rename files such that file name as numbers with starting name as file string to filter files in a given folder, such as model_train_1.py, moldel_train_2.py... Here model_train_1.py is lowest resoluton file. Each file half the volume of the preceeding file, where number of particles and cells are kept the same. Default code take three simulations. 

``` python
    # Save multiple named arrays to the same file
    # each field shape  # of examples x # pixels
    data_dict = {'density': density, 'temp': temp, 'vpec': vpec, 'nHI' : nHI}

    save_file = dir_output+'model_test.npy'
    print('saving ', save_file)
    with open(save_file, 'wb') as f:
          np.savez(f, **data_dict)
```



## batch script to run on HPC raven
```command
#!/bin/bash -l
#SBATCH -o ./out.%j
#SBATCH -e ./out.err.%j
#SBATCH -D ./
#SBATCH -J train_gpu
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000

#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=12:00:00

module purge
module load intel/21.3.0
module load anaconda/3/2021.11
module load tensorflow/cpu/2.9.2
module load keras-preprocessing/1.1.2
module load cuda/11.6
module load cudnn/8.4.1
module load tensorflow/gpu-cuda-11.6/2.9.0
module load keras/2.9.0
module load tensorboard/2.8.0
module load tensorflow-probability/0.14.1

srun -u python main.py --epochs 1000 --latent True
```

