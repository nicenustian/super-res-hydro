import matplotlib
import numpy as np
import pickle
import matplotlib.pyplot as plt

font = {'family' : 'serif', 'weight' : 'normal','size' : 34}
matplotlib.rc('font', **font)    
colors_lines = ['black', 'red', 'orange', 'blue', 'purple', 'magenta']


def plot_pdf(fig, ax, keys_list, pp, linestyle='-', alpha=1., color_usr=None):
    
    x_values, pdf = pp
    num_features = pdf.shape[0]

    for fi in range(num_features):
                
        current_ax = ax[fi] if num_features > 1 else ax
        
        current_ax.text(0.01, 0.85, keys_list[fi],
                        transform=current_ax.transAxes, 
                        color=colors_lines[fi])
        
        current_ax.plot(x_values[fi], pdf[fi], color=color_usr, 
                        linestyle=linestyle, alpha=1.)
        
        current_ax.set_xlabel(r'${\rm X}$')
        current_ax.set_ylabel(r'${\rm PDF}$')
        current_ax.set_xlim(-.1, 1.0)
        
        if num_features > 1 and fi < num_features - 1:
            current_ax.set_xticklabels([])



def plot_ps(fig, ax, keys_list, pp, num_features, linestyle='-', alpha=1., 
            label=None, color_usr=None):
    k, ps = pp
 
    for fi in range(num_features):
        current_ax = ax[fi] if num_features > 1 else ax
                
        current_ax.text(0.01, 0.85, keys_list[fi], 
                        transform=current_ax.transAxes, color=colors_lines[fi])
        
        current_ax.plot(k[fi], ps[fi], color=color_usr, linestyle=linestyle, 
                        alpha=alpha, label=label)
        
        #current_ax.set_xscale('log')
        #current_ax.set_yscale('log')
        #current_ax.set_ylim(10, 1e4)
        
        current_ax.set_xlabel(r'$k/{\rm [Mpc/h]}$')
        current_ax.set_ylabel(r'${\rm kp(k)}$')
        
        if num_features > 1 and fi < num_features - 1:
            current_ax.set_xticklabels([])
        
        current_ax.tick_params(which='both', direction="in", width=1.5)
        current_ax.tick_params(which='major', length=14, top=True, left=True, 
                               right=True)
        current_ax.tick_params(which='minor', length=10, top=True, left=True, 
                               right=True)
        current_ax.minorticks_on()
        
        if label is not None:
            current_ax.legend(frameon=False, fontsize=18, handletextpad=0.1,
                              loc='lower left', handlelength=1)
        
    
    
def plot_skewers(
        output_dir, epoch, quantities, 
        data, fake, box_sizes, 
        num_features, num_img, latent
        ):
    
    if not latent:
        batch_size = fake[1].shape[0]
    else:
        batch_size = fake[0].shape[0]
        
    boxes = len(box_sizes)
    biggest_box_size = np.max(box_sizes)
    biggest_box_factor = np.int32(biggest_box_size/np.min(box_sizes))


    for qi, quantity in enumerate(quantities):
    
        fig, ax = plt.subplots(num_img*boxes*2, 1, figsize=(28, 4*num_img*boxes*2))      
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        
        index = np.random.choice(data[0].shape[0]-biggest_box_factor, num_img, replace=False)
        index_fake = np.random.choice(batch_size, num_img, replace=False)
    
        for bi, box_size in enumerate(box_sizes):
            for i, (ireal, ifake) in enumerate(zip(index, index_fake)):
                    
                    plot_index = num_img * boxes + i*boxes + bi
                    box_factor = np.int32(biggest_box_size/box_size)
                                        
                    ax[plot_index].text(0.01, 0.85, 'real '+quantity,
                                        transform=ax[plot_index].transAxes, 
                                        color=colors_lines[bi])
                        
                    axis = np.arange(data[bi].shape[1]*box_factor)/(data[bi].shape[1]*box_factor)*box_sizes[bi]
                    
                    ax[plot_index].text(0.01, 0.6, str(np.int32(box_sizes[bi]))+'-'+str(data[bi].shape[1]), 
                                        transform=ax[plot_index].transAxes, size=24,
                                        color=colors_lines[bi])
                        
                    
                    ax[plot_index].step(axis, np.reshape(data[bi][ireal:ireal+box_factor,:, qi], -1), 
                            color=colors_lines[bi], alpha=0.8)
                    ax[plot_index].set_xlim(np.min(axis), np.max(axis))
                    
                    ax[plot_index].set_yticks([])
                    ax[plot_index].set_xticks([])
    
                    
                    if bi<len(fake):
                                        
                        plot_index = i*boxes + bi                        
                        ax[plot_index].text(0.01, 0.85, 'gen. '+quantity,
                                        transform=ax[plot_index].transAxes, 
                                        color=colors_lines[bi])
                        
                        
                        axis = np.arange(fake[bi].shape[1])/fake[bi].shape[1]*box_sizes[0]
                        ax[plot_index].text(0.01, 0.6, str(np.int32(box_sizes[0]))+'-'+str(fake[bi].shape[1]), 
                                                transform=ax[plot_index].transAxes, size=24,
                                              color=colors_lines[bi])
                        
                        ax[plot_index].step(axis, fake[bi][ifake,:, qi], 
                            color=colors_lines[bi], alpha=0.8)
                        ax[plot_index].set_xlim(np.min(axis), np.max(axis))
                      
                    
                    ax[plot_index].set_yticks([])
                    ax[plot_index].set_xticks([])
                    
        
        plt.tight_layout()
        plt.savefig(output_dir+quantity+'_skewers_epoch'+str(epoch+1)+'.jpg')
        plt.close()



# Plot training history
def plot_training_history(output_dir):

    fig, ax = plt.subplots(3,1, figsize=(10, 18))
    fig.subplots_adjust(wspace=0., hspace=0.)
    
    min_x = 0    
    lw = 2
    
    #reading the best model, history and data files
    history = pickle.load( open(output_dir+'history', 'rb') )
    epochs = np.arange(0, len(history['g_loss']))
    epochs = epochs[min_x:]
    g_loss = np.array(history['g_loss'][min_x:])
    d_loss = np.array(history['d_loss'][min_x:])
    p_loss = np.array(history['p_loss'][min_x:])
    pdf_loss = np.array(history['pdf_loss'][min_x:])
 
    # Plot training & validation accuracy values
    ax[0].plot(epochs, g_loss, color='black', linewidth=lw, label='Generator')
    ax[0].plot(epochs, d_loss, color='red', linewidth=lw, label='Discriminator')

    ax[0].set_ylabel('Gen./Dis.')
    ax[0].set_xlabel('')
    ax[0].axhline(0, color='orange', alpha=0.5, linewidth=lw)
    #ax[0].set_yscale('log')
    ax[0].set_xlim(min_x,)
    
    ax[0].set_xticklabels([])
    
    ax[1].plot(epochs, p_loss, color='black', linewidth=lw, label='kp(k)')
    ax[1].set_ylabel('kp(k)')
    ax[1].set_xlabel('')
    ax[1].set_xlim(min_x,)    
    #ax[1].set_xlim(min_x, max_x)
    #ax[1].set_ylim(min_y, max_y)
    ax[1].set_xticklabels([])
    
    
    ax[2].plot(epochs, pdf_loss, color='black', linewidth=lw, label='PDF')
    ax[2].set_ylabel('PDF')
    ax[2].set_xlabel('Epochs')
    ax[2].set_xlim(min_x,)
    #ax[0].set_ylim(min_y, max_y)
    #ax[2].legend(frameon=False)
    
    for pi in range(3):
        ax[pi].tick_params(which='both',direction="in", width=1.5)
        ax[pi].tick_params(which='major',length=14,top=True,left=True,right=True)
        ax[pi].tick_params(which='minor',length=10, top=True,left=True,right=True)
        ax[pi].minorticks_on()
    
    fig.savefig(output_dir+'training.pdf',format='pdf', 
                dpi=90, bbox_inches = 'tight')
