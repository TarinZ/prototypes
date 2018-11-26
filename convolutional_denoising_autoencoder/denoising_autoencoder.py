from helper_functions import *
# import torchvision.datasets as dset



def select_signals(index, signal_batch, noisy_signal_batch, output_batch):    
    sig = np.array(signal_batch)[index,:]
    noisy_sig = np.array(noisy_signal_batch)[index,:]
    out = np.array(output_batch.detach().numpy())[index,:]
    return [sig, noisy_sig, out]


def show_denoised(ax, sig_noisysig_out, ylims = [-1.5, 1.5], pause_time = 0.0, save_flag = False):    
    
    # Special graph for viz
    if save_flag == True:
        for ii in range(2):
            ax[ii].cla()       
            ax[ii].plot(sig_noisysig_out[ii+1].T,'-g')
            ax[ii].set_ylim(ylims)        
            plt.show()
        # plt.tight_layout()            

    else:
        for ii in range(3):
            ax[ii].cla()       
            ax[ii].plot(sig_noisysig_out[ii].T,'.-')
            ax[ii].set_ylim(ylims)
            plt.show()
    if pause_time > 0.0:            
        plt.pause(pause_time)


def create_signals(n_data_points, n_total, f_fs, f_range, t_range):
    original_signals = np.zeros((n_data_points, n_total)).astype(np.float32)
    for ii in range(n_data_points):    
        # signal parameters
        f_signal = np.random.uniform(f_range[0], f_range[1])
        t_signal = np.random.uniform(t_range[0], t_range[1])
        n_signal = np.int(np.round(t_signal * f_fs))
        i_begin = np.int(np.floor(np.random.uniform(0, n_total - n_signal)))

        # create the signal    
        signal = np.array([np.sin(2*np.pi*f_signal*nn/f_fs) for nn in range(n_signal)]).astype(np.float32)            
        original_signals[ii, i_begin : i_begin + n_signal] = signal
    return original_signals



# Set Data Loader(input pipeline)
N = lambda:0
F = lambda:0
T = lambda:0
I = lambda:0


# User inputs
parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('-s', dest='save_denoising_images_directory', required = False, type=str)
parser.add_argument('-g', dest='gpu', required = False, type=int)
args = parser.parse_args()

# Plots
fig1, ax1 = plt.subplots(1,3, figsize = (15,5))
fig2, ax2 = plt.subplots(1,2)
if args.save_denoising_images_directory:

    # Create the figure showing the denoising
    fig3, ax3 = plt.subplots(1,2, figsize=(10,5))
        
    # Create the directory if it doesn't exist.        
    if not os.path.exists(args.save_denoising_images_directory):
        os.makedirs(args.save_denoising_images_directory)
    
plt.pause(0.01)

# pdb.set_trace()


N.training_data_points = 1000
N.val_data_points = 50
F.fs = 5000.0
F.range = [400, 800]
T.range = [0.05, 0.3]
T.total = 4096/5000.0
N.total = np.int(np.round(F.fs * T.total)) # samples
noise_var = 0.5
# original_signals = np.zeros((N.data_points, N.total)).astype(np.float32)
# Set Hyperparameters
N.epochs = 200
N.batch_size = 10
N.batches_per_val = 15 # The validation set is run every N.batches_per_val batch times.
# N.batches_per_val = 10 # The validation set is run every N.batches_per_val batch times.
# learning_rate = 0.0003
learning_rate = 0.00005
N.batches_per_epoch = int(np.floor(N.training_data_points / float(N.batch_size)))



training_signals = create_signals(N.training_data_points, N.total, F.fs, F.range, T.range);
training_signals = np.expand_dims(training_signals, 1)
# t_training_signals = torch.Tensor(training_signals)

val_signals = create_signals(N.val_data_points, N.total, F.fs, F.range, [0.35, 0.5 ]);
val_signals = np.expand_dims(val_signals, 1)


train_loader = torch.utils.data.DataLoader(dataset=torch.Tensor(training_signals), batch_size = N.batch_size , shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=torch.Tensor(val_signals), batch_size = N.val_data_points , shuffle=False)
# pdb.set_trace()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(1,8,3,stride = 1, padding=1, bias = True),                        
                        nn.ReLU(),
                        nn.BatchNorm1d(8),
                        nn.MaxPool1d(2,2),
                        nn.Conv1d(8,16,3,stride = 1, padding=1, bias = True),                        
                        nn.ReLU(),
                        nn.BatchNorm1d(16),
                        nn.MaxPool1d(2,2),
                        nn.Conv1d(16,32,3,stride = 1, padding=1, bias = True),                    
                        nn.ReLU(),
                        nn.BatchNorm1d(32),
                        nn.MaxPool1d(2,2),
                        nn.Conv1d(32,64,3,stride = 1, padding=1, bias = True),                        
                        nn.ReLU(),
                        nn.BatchNorm1d(64),
                        nn.MaxPool1d(2,2)
        )
                    
    def forward(self,x):
        
        out = self.layer1(x)                
        return out


# If GPU in use
if args.gpu == 1:    
    print("GPU")
    encoder = Encoder().cuda()
else:
    encoder = Encoder()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(                        
                        nn.ConvTranspose1d(64, 32, 3, stride = 2, padding = 1, output_padding = 1, bias = True),                         
                        nn.Tanh(),
                        nn.BatchNorm1d(32),
                        nn.ConvTranspose1d(32, 16, 3, stride = 2, padding = 1, output_padding = 1, bias = True), 
                        nn.Tanh(),
                        nn.BatchNorm1d(16),
                        nn.ConvTranspose1d(16, 8, 3, stride = 2, padding = 1, output_padding = 1, bias = True), 
                        nn.Tanh(),
                        nn.BatchNorm1d(8),
                        nn.ConvTranspose1d(8, 1, 3, stride = 2, padding = 1, output_padding = 1, bias = True),
                        nn.Tanh()
        )
        
        
    def forward(self,x):                
        out = self.layer1(x)                
        return out


if args.gpu == 1:
    print("GPU")
    decoder = Decoder().cuda()
else:    
    decoder = Decoder()


parameters = list(encoder.parameters())+ list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

I.batches_run_so_far = 0
accumulated_training_loss = 0.0


# Loss vectors
loss_vector = np.zeros((3,N.batches_per_epoch*N.epochs)).astype(np.float32)

N.total_times_val_run = np.int(np.ceil((N.epochs * N.batches_per_epoch) / float(N.batches_per_val))) # Total number of times the validation set is run
train_val_loss_vector = np.zeros((2, N.total_times_val_run)).astype(np.float32)
# pdb.set_trace()
I.incrementor = 0

# Begin training
for ee in range(N.epochs):


    # Initialize the iterator. 
    training_batch_iterator = iter(train_loader)    
    

    for bb in range(N.batches_per_epoch):

        # Get signal batch
        signal = training_batch_iterator.next()
        
        # Create a noise vector
        noise = torch.Tensor(noise_var*np.random.randn(N.batch_size, 1, N.total))
        
        # Add noise to the signal
        noisy_signal = signal + noise

        # Get signal and noise_signal as variables        
        if args.gpu == 1:
            signal = Variable(signal).cuda()
            noisy_signal = Variable(noisy_signal).cuda()

        
        # Zero out the gradients so that they are not accumulated
        optimizer.zero_grad()

        # Pass through denoising autoencoder
        encoder_output = encoder(noisy_signal)
        output = decoder(encoder_output)

        # Compute loss
        loss = loss_func(output,signal)

        # Perform the backward pass
        loss.backward()

        # Update the weights
        optimizer.step()            

        # Save loss for this batch        
        loss_vector[0, ee*N.batches_per_epoch + bb] = float(loss.data)

        # Accumulate training loss
        accumulated_training_loss += float(loss.data)

        # If time to process validation:
        if (np.mod(I.batches_run_so_far, N.batches_per_val) == 0):
            
            # Store the accumulated training loss            
            train_val_loss_vector[0,I.incrementor] = accumulated_training_loss / float(N.batches_per_val)

            # Reset the accumulated training loss
            accumulated_training_loss = 0.0

            # Process validation set
            val_batch_iterator = iter(val_loader)

            # Place DNN back into evaluation mode (freezes batchnorm settings, etc)
            encoder.eval()
            decoder.eval()

            # pdb.set_trace()
            val_signal = val_batch_iterator.next()
            val_noise = torch.Tensor(noise_var * np.random.randn(N.val_data_points, 1, N.total))
            val_noisy_signal = val_signal + val_noise                        
            if args.gpu == 1:
                val_signal = Variable(val_signal).cuda()
                val_noisy_signal = Variable(val_noisy_signal).cuda()
                                                

            # Pass through denoising autoencoder
            # val_output = decoder(encoder(val_noisy_signal))
            val_encoder_output = encoder(val_noisy_signal)

            # pdb.set_trace()
            
            val_output = decoder(val_encoder_output)
            
            # Compute val loss
            val_loss = loss_func(val_output, val_signal)

            # Store off the validation loss
            train_val_loss_vector[1,I.incrementor] = float(val_loss.data)

            # Place DNN back into training mode
            encoder.train()
            decoder.train()

            # Increment the index for train_val_loss_vector
            I.incrementor += 1


            # Show accumulated training loss every validation time
            ax2[1].cla()
            ax2[1].plot(train_val_loss_vector[0,:].T,'.-'); 
            ax2[1].plot(train_val_loss_vector[1,:].T,'.-g'); plt.show()
            ax2[1].set_ylim([-0.05, 0.3])
            ax2[1].grid(True)


            
            
        # accumulate the count for the number of batches run thus far
        I.batches_run_so_far += 1
        
        
    

    # Print stats
    print("Epoch: %1d / %1d" % (ee, N.epochs) , " MSE loss: %2.4f" % float(loss.data))    

    if args.gpu == 1:
        val_signal = val_signal.cpu()
        val_noisy_signal = val_noisy_signal.cpu()
        val_output = val_output.cpu()
        signal = signal.cpu()
        noisy_signal = noisy_signal.cpu()
        output = output.cpu()

    # show_denoised(ax1, select_signals(0, val_signal, val_noisy_signal, val_output), ylims = [-2, 2])
    show_denoised(ax1, select_signals(0, signal, noisy_signal, output), ylims = [-2, 2])

    # Show every iteration of training loss
    ax2[0].cla()
    ax2[0].plot(loss_vector[0,:].T,'.-'); plt.show()
    ax2[0].set_ylim([-0.05, 0.25])
    ax2[0].grid(True)

    
    # Show result of denoising for this epoch
    if args.save_denoising_images_directory:
        show_denoised(ax3, select_signals(0, val_signal, val_noisy_signal, val_output), ylims = [-2, 2], save_flag = True)
        
        
        filename = "image" + (str(ee) + ".png").zfill(8)        
        # pdb.set_trace()
        fig3.savefig(args.save_denoising_images_directory + "/" + filename)
        


    # If saving is occuring, plot and save the image
    # show_denoised(ax3, select_signals(0, val_noisy_signal, val_output), ylims = [-3.5, 3.5], )
    



    plt.pause(0.001) 

# check image with noise and denoised image\
print("OK")
pdb.set_trace()




# I.selected = 7
# sig = np.array(signal)[I.selected,:]
# noisy_sig = np.array(noisy_signal)[I.selected,:]
# out = np.array(output.detach().numpy())[I.selected,:]

# fig, ax = plt.subplots(1,3, figsize = (15,5))
# ax[0].plot(sig.T,'.-')
# ax[1].plot(noisy_sig.T,'.-')
# ax[2].plot(out.T,'.-')
# plt.show()


# pdb.set_trace()


##########

# sig = signal[7,:] + signal[12,:]
# nsig = sig + torch.Tensor(0.1*np.random.randn(1, 1, N.total))
# encoder.eval()
# decoder.eval()
# sig_out = decoder(encoder(nsig)).detach().numpy()
# fig, ax = plt.subplots(1,3, figsize = (15,5))
# ax[0].plot(np.array(sig).T,'.-')
# ax[1].plot(np.array(nsig[0]).T,'.-')
# ax[2].plot(sig_out.T,'.-')
# plt.show()


# img = image[0].cpu()
# input_img = image_n[0].cpu()
# output_img = output[0].cpu()

# origin = img.data.numpy()
# inp = input_img.data.numpy()
# out = output_img.data.numpy()

# plt.imshow(origin[0],cmap='gray')
# plt.show()

# plt.imshow(inp[0],cmap='gray')
# plt.show()

# plt.imshow(out[0],cmap="gray")
# plt.show()

# print(label[0])
