import matplotlib.pyplot as plt

def plot_fom_data():
     """
     Look in directories in this folder for files named fom.npy in subfolder
     images and record its mean and std. In the subfolder wandb/run*/files
     open config.yaml and record the value of n_samples. Plot the means and std
     against n_samples on x-axis.·
     """
     means = []
     stds = []
     n_samples = []

     subdirs = find_fom(".")

     for fom_path in subdirs:
         subdir = os.path.dirname(fom_path)
         config_file = find_config(subdir)
         if not config_file:
             continue
         fom = np.load(fom_path)
         means.append(np.mean(fom))
         stds.append(np.std(fom))
         n_samples.append(get_n_samples_from_config(config_file[0]))

     # sort n_samples, means, and stds by n_samples
     n_samples, means, stds = zip(*sorted(zip(n_samples, means, stds)))
     plt.errorbar(n_samples, means, yerr=stds, fmt='o')
     plt.xlabel("Number of Samples")
     plt.ylabel("FOM Mean ± Std")
     plt.box(False)
     plt.tight_layout()
     plt.savefig("fom.png")
     plt.close()
