# super-res-hydro
Generate super resolution one dimensional fields using Multiscale Wasserstein Generative Adversarial Network with Gradient Penalty (MS-WGAN-GP) with converging statistical properties such as power spectrum and PDF from outputs of Hydrodynamical simulations 

# Problem Statement

The Hydrodynamical Simulations are limited in terms of number of particles/cells can used to represent a part of Universe. One way is to simulate different volumes of Universe with same number of particles. Later, using MS-WGAN-GP to paint the properties of smaller volumes onto larger all with same cells/particles.  However, we need to have these fields with similar statistical properties at different volumnes. Therefore, we empoly statistical metrics such as power spectrum and PDF to converge at different resolutions. Below we have shown the convergence of these metrics and the qunatities during training (1000 epochs). The training data is from simulations with Volumes 160Mpc/h, 80Mpc/h and 40Mpc/h all with same particles of 2048^3.  


```command
python main.py --epochs 1000 --lr 1e-4 --output_dir ml_output --dataset_dir dataset 
```

## Power spectrum of one dimensional fields
https://github.com/nicenustian/super-res-hydro/assets/111900566/f58a51c7-0264-4c58-b20f-b60ff61c842b

## PDF of one dimensional fields
https://github.com/nicenustian/super-res-hydro/assets/111900566/16708ebb-2e42-42db-b7e7-999679c8e983

## velocities
https://github.com/nicenustian/super-res-hydro/assets/111900566/6c23eb39-de89-4ecd-9542-f5f8b08e7fde

## gas density field
https://github.com/nicenustian/super-res-hydro/assets/111900566/c82eb02e-63f9-489b-ae31-91c431557392

## temp field
https://github.com/nicenustian/super-res-hydro/assets/111900566/c34d61ac-4429-4ee4-af27-c72eb59fbb03



Please provide the one dimensional fields for each simulation in dataset_dir folder. Each file should be either .npy or .hdf5. The field name's should be output as dictionary, see example below. Each file is simulation ran at particular volume. Please make sure the code reads them in order of lowest to highest resolution. THe order of fields read is displayed in the start. If it is not read in order rename files such that file name as numbers with starting name as file string to filter files in a given folder, such as model_train_1.py, moldel_train_2.py... Here model_train_1.py is lowest resoluton file. Eeach file half the volume of the preceeding file, where number of particles and cells are kept the same. Default code take three simulations. 

```python
    # Save multiple named arrays to the same file
    # each field shape  # of examples x # pixels
    data_dict = {'density': density, 'temp': temp, 'vpec': vpec, 'nHI' : nHI}

    save_file = dir_output+'model_test.npy'
    print('saving ', save_file)
    with open(save_file, 'wb') as f:
          np.savez(f, **data_dict)
```
