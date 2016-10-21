import os
import ntpath
import matplotlib.pylab as plt
import numpy as np
import kaldi_nnet_tools as knt
import seaborn as sne
plt.ion()
# path where model info nnet-am-info and copy nnet-am-copy are saved
am_copy_path = 'models/fisher_final.mdl.nnet.txt'
am_info_path = 'models/fisher_final.mdl.info.txt'

# convert kaldi model to python
net = knt.parseNNET(am_copy_path, am_info_path)

# read context and iVector(constant component) dimension
left_context = abs(net[2]['<Context>'][0])
right_context = abs(net[2]['<Context>'][-1])
const_component_dim = net[2]['<ConstComponentDim>']


# learn context-dependent states of the word help
folder_path = "/Users/rafaelvalle/Desktop/pasr/features/help/"
filepaths_help = [folder_path + filename for filename in os.listdir(folder_path)
                  if (os.path.isfile(os.path.join(folder_path, filename))
                  and not filename.startswith('.'))]

outputs = None
for feature_path in filepaths_help:
    # read kaldi features and transform them into numpy array
    data = knt.read_kaldi_features(feature_path)

    # create generator with spliced data and iVectors
    data_gen = knt.splice(data, left_context, right_context, const_component_dim)

    # compute ouputs using neural network layers
    output = [knt.forward(i, net[3:], verbose=False) for i in data_gen]
    output = np.clip(output, 1.0e-20, np.inf)
    if outputs is None:
        outputs = output
    else:
        outputs = np.vstack((outputs, output))

# compute summary statistics of the word context-dependent states
means = outputs.mean(axis=0)
percentiles = (50, 60, 70, 80, 90, 95)

# save log likelihoods with full and sparse state vector
folder_path = "/Users/rafaelvalle/Desktop/pasr/features/other/"
filepaths_other = [
    folder_path + filename for filename in os.listdir(folder_path)
    if (os.path.isfile(os.path.join(folder_path, filename)) and not
        filename.startswith('.'))]


folder_path = "/Users/rafaelvalle/Desktop/pasr/features/help_plus/"
filepaths_help_plus = [
    folder_path + filename for filename in os.listdir(folder_path)
    if (os.path.isfile(os.path.join(folder_path, filename)) and not
        filename.startswith('.'))]

for feature_path in (filepaths_help + filepaths_help_plus + filepaths_other):
    # read kaldi features and transform them into numpy array
    data = knt.read_kaldi_features(feature_path)

    # create generator with spliced data and iVectors
    data_gen = knt.splice(
        data, left_context, right_context, const_component_dim)

    # compute ouputs using neural network layers
    output = [knt.forward(i, net[3:], verbose=False) for i in data_gen]
    output = np.clip(output, 1.0e-20, np.inf)

    knt.save_kaldi_loglikelihoods(
        np.log(output),
        "log_likelihoods/{}.ark".format(filename))

    for percentile in percentiles:
        ids_filter = means < np.percentile(means, percentile)

        # alter data
        output_altered = output.copy()
        output_altered[:, ids_filter] = output_altered.min()

        # save logikelihods
        filename = os.path.splitext(ntpath.basename(feature_path))[0]

        knt.save_kaldi_loglikelihoods(
            np.log(output_altered),
            "log_likelihoods/{}_{}_altered.ark".format(filename, percentile))

    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 6))
    im1 = axes[0].imshow(means.T,
                        interpolation='nearest',
                        origin='low',
                        aspect='auto',
                        cmap=plt.cm.Oranges)
    plt.colorbar(im1, ax = axes[0])

    im2 = axes[1].imshow(stds.T,
                        interpolation='nearest',
                        origin='low',
                        aspect='auto',
                        cmap=plt.cm.Oranges)
    plt.colorbar(im2, ax = axes[1])

    im3 = axes[2].imshow(outputs,
                        interpolation='nearest',
                        origin='low',
                        aspect='auto',
                        cmap=plt.cm.Oranges)
    plt.colorbar(im3, ax = axes[2])

    im4 = axes[3].imshow(outputs_altered,
                        interpolation='nearest',
                        origin='low',
                        aspect='auto',
                        cmap=plt.cm.Oranges)
    plt.colorbar(im4, ax = axes[3])
    """
