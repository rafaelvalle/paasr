import subprocess
import re
import numpy as np


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def parseNNET(am_copy_path, am_info_path):
    """Converts a kaldi neural network into a python structure by using
    non-binary files generated using kaldi's nnet-am-copy and nnet-am-info
    PARAMS
    ------
    am_copy_path : str
        Path to text model file generated with nnet-am-copy
    am_info_path
        Path to text model file generated with nnet-am-info
    RETURNS
    -------
    net : list
        ANN model
    """

    print 'Parsing nnet-am-copy'
    with open(am_copy_path, 'r') as f:
        ann = f.read().replace('\n', ' ')
        regex = ur"</[a-zA-Z]+>"
        components = ['<'+n[2:] for n in re.findall(regex, ann)]
        ann = ann.split()
    items = []
    idx_node = -1
    i = 0
    last_tag = False
    while i < len(ann):
        if ann[i] == ('</Nnet>'):  # is last tag?
            last_tag = True
        elif ann[i].startswith('</'):  # is end of component
            print 'Completed component {}'.format(ann[i])
        elif ann[i].startswith('<'):  # is component or attribute?
            if ann[i] in components:  # is component?
                print '\nStarted component {}'.format(ann[i])
                items.append({'<Name>': ann[i]})
                idx_node += 1
            else:
                print 'Started attribute or feature {}'.format(ann[i])
                cur_feat = ann[i]
                items[idx_node][cur_feat] = []
        elif ann[i] not in ('[', ']'):
            items[idx_node][cur_feat].append(float(ann[i]))
        elif ann[i] == '[' and last_tag:
            print 'Started non-tagged attribute <Priors>'
            items.append({'<Name>': '<Priors>'})
            idx_node += 1
            cur_feat = '<Values>'
            items[idx_node][cur_feat] = []
        i += 1

    # convert lists with one item to single numbers and arrays to numpy arrays
    for i in items:
        for k in i.keys():
            if k == '<Context>':
                i[k] = np.array(i[k], dtype=int)

            elif not isinstance(i[k], str):
                if len(i[k]) == 1:
                    if i[k][0].is_integer():
                        i[k] = int(i[k][0])
                    else:
                        i[k] = float(i[k][0])
                else:
                    i[k] = np.array(i[k])

    print '\nParsing nnet-am-info'
    # assumes that the order is the same in nnet-am-copy and nnet-am-info
    # and that the first index is the NNET and the second one instantiates
    # the components tag
    idx_component = 1
    for line in tuple(open(am_info_path, 'r')):
        line = line.replace(' = ', '=')
        if line.startswith('component'):
            print line
            for i in line.split():
                if 'Component' in i:
                    idx_component += 1
                    cur_feat = '<{}>'.format(i.replace(',', ''))
                    print 'Parsing Input and Output dimensions of {}'.format(cur_feat)
                elif i.startswith('input-dim'):
                    print 'Updating <InputDim> on {}'.format(cur_feat)
                    items[idx_component]['<InputDim>'] = int(i.split('=')[1].replace(',', ''))
                elif i.startswith('output-dim'):
                    print 'Updating <OutputDim> on {}'.format(cur_feat)
                    items[idx_component]['<OutputDim>'] = int(i.split('=')[1].replace(',', ''))

    print "\nSetting shape of Linear and Bias params"
    for i in items:
        if ('Component' in i['<Name>'] and '<InputDim>' in i and '<OutputDim>' in i
                and i['<Name>'] not in (
                    '<NormalizeComponent>', '<PnormComponent>', '<SpliceComponent>', '<SoftmaxComponent>',
                    '<SumGroupComponent>')):
            print 'Param {}'.format(i['<Name>'])
            i['<LinearParams>'] = i['<LinearParams>'].reshape(i['<InputDim>'], i['<OutputDim>'])
            i['<BiasParams>'] = i['<BiasParams>'].reshape(1, i['<BiasParams>'].shape[0])

    return items


def p_norm(data, size, p):
    """(\sum(\abs(Xi) ^ p) ^ 1/p)
    PARAMS
    ------
        size: int
            The group size for dimensionality reduction
        p: int
            Norm
    RETURNS
    -------
        P-Normed data
    """

    output = np.power(np.sum(np.power(np.abs(np.reshape(
            data, (data.shape[1]/size, size))), p), axis=0), 1.0/p)
    return output.reshape(1, output.shape[0])


def normalization_nonlinearity(data):
    """Fixed non-trainable non-linearity to renormalize the data to
    have unit standard deviation. Used in kaldi's nnet2 model.
    PARAMS
    ------
    data : np.array
        Data to apply normalization nonlinearity

    RETURNS
    -------
        Normalized data
    """
    sigma = np.sqrt(np.average(data ** 2))
    if sigma > 1:
        data = data * (1.0/sigma)
    return data


def softmax(data):
    """Computes the softmax of input data
    PARAMS
    ------
    data : array
        Data to apply normalization nonlinearity
    RETURNS
    -------
        Softmax of input data
    """
    return np.exp(data) / np.exp(data).sum()


def sum_group(data, groups):
    """Groups data according to groups. This can be used in the output layer of
    neural networks to combine results. It is used in kaldi's nnet2.
    PARAMS
    ------
    data : array
        Data to apply normalization nonlinearity
    groups : list
        Sequence of group sizes
    RETURNS
    -------
        Grouped data
    """
    cumsum = groups.cumsum().astype(int)
    return np.array([np.sum(data[:, :groups[1]])] + [np.sum(data[:, p[0]:p[1]])
                    for p in np.stack((cumsum[:-1], cumsum[1:])).T - 1])


def compute_linear(data, layer):
    """Computes the dot product of data and weights and sums bias given a layer
    of a network created with parsetNNET.
    PARAMS
    ------
    data : array
        Data to apply normalization nonlinearity
    layer : dictionary
        Layer created with parseNNET
    RETURNS
    -------
        Linear combination
    """
    return np.dot(data, layer['<LinearParams>']) + layer['<BiasParams>']


def predict(data, layers, per_layer=False, verbose=False):
    """Predicts the output of the network given data
    PARAMS
    ------
    data : array
        Data to apply normalization nonlinearity
    layer : list of dics
        List of network layers created with parseNNET
    verbose: boolean
        Describe which computation is being performed
    RETURNS
    -------
    prediction : list
        Prediction for each frame in data
    """
    output = data
    outputs = []
    for i in xrange(len(layers)):
        if 'Component' in layers[i]['<Name>']:
            if layers[i]['<Name>'] == '<NumComponents>':
                continue
            elif layers[i]['<Name>'] == '<PnormComponent>':
                output = p_norm(
                    output, layers[i]['<OutputDim>'], layers[i]['<P>'])
            elif layers[i]['<Name>'] == '<NormalizeComponent>':
                output = normalization_nonlinearity(output)
            elif layers[i]['<Name>'] == '<SoftmaxComponent>':
                output = softmax(output)
            elif layers[i]['<Name>'] == '<SumGroupComponent>':
                output = sum_group(output, layers[i]['<Sizes>'])
            else:
                output = compute_linear(output, layers[i])
            if verbose:
                print layers[i]['<Name>'], output.shape
            if per_layer:
                outputs.append(output)
    if per_layer:
        return outputs
    return output


def extract_kaldi_features(wav_dump_features_path, options, spk2utt_rspecifier,
                           wav_rspecifier, feature_wspecifier):
    """Python wrapper for Kaldi's online online2-wav-dump-features
    online2-wav-dump-features [options] <spk2utt-rspecifier> <wav-rspecifier>
    <feature-wspecifier>
    PARAMS
    ------
    wav_dump_features_path : str
        Check Kaldi's website for description
    options : str
        Check Kaldi's website for description
    spk2utt_rspecifier : str
        Check Kaldi's website for description
    wav_rspecifier : str
        Check Kaldi's website for description
    feature_wspecifier : str
        Check Kaldi's website for description
    RETURNS
    -------
    data : np.array
        Features saved as described in feature_wspecifier
    """
    return None  # not implemented yet
    cmd = '{} {} "{}|" "{}|" "{}"'.format(
        wav_dump_features_path, options, spk2utt_rspecifier, wav_rspecifier,
        feature_wspecifier)
    subprocess.Popen(cmd)

    return read_kaldi_features(feature_wspecifier.split(":")[1])


def read_kaldi_features(filepath):
    """Read features saved using Kaldi's online online2-wav-dump-features
       PARAMS
       ------
           filepath : str
               filepath of file to be loaded
       RETURNS
       -------
           data : np.array
               Features saved as described in feature_wspecifier
    """

    feats = []
    # slower but safer
    for line in tuple(open(filepath, 'r')):
        cur_frame = []
        for i in ' '.join(line.split()).split():
            try:
                cur_frame.append(float(i))
            except:
                print "{} is not numeric".format(i)
        if len(cur_frame) > 0:
            feats.append(cur_frame)
        else:
            print "Ignoring {}".format(line)

    return np.array(feats, ndmin=2)


def splice(data, left_context, right_context, const_component_dim):
    prefix = np.zeros((left_context, data.shape[1]))
    suffix = np.tile(data[-1], (7, 1))
    data = np.concatenate((prefix, data, suffix))

    i = left_context
    while i < len(data) - right_context:
        feat = data[i-left_context:i+right_context+1]
        feat = np.concatenate(
            (feat[:, :data.shape[1]-const_component_dim].ravel(),
             feat[:, -const_component_dim:].mean(axis=0)))
        i += 1
        yield feat

    """
    # though good memory complexity, has very bad time complexity
    # might need fixes
    while i < len(data):
        l_idx = i - left_context
        r_idx = i + right_context + 1
        feat = data[max(0, l_idx): min(len(data), r_idx)]
        if l_idx < 0:
            prefix = np.zeros((abs(l_idx), data.shape[1]))
            feat = np.concatenate((prefix, feat))
        if r_idx > len(data) - 1:
            set_trace()
            suffix = np.tile(data[-1], (r_idx - (len(data) - 1), 1))
            feat = np.concatenate((feat, suffix))
        feat = np.concatenate(
            (feat[:, :data.shape[1]-const_component_dim].ravel(),
             feat[:, -const_component_dim:].mean(axis=0)))
        i+= 1

        yield feat.reshape(1, len(feat))
        """


# path where model info nnet-am-info and copy nnet-am-copy are saved
am_copy_path = 'models/fisher_final.mdl.nnet.txt'
am_info_path = 'models/fisher_final.mdl.info.txt'

# convert kaldi model to python
net = parseNNET(am_copy_path, am_info_path)

# read context and iVector(constant component) dimension
left_context = abs(net[2]['<Context>'][0])
right_context = abs(net[2]['<Context>'][-1])
const_component_dim = net[2]['<ConstComponentDim>']

"""
# TODO : extract features from python
wav_dump_features_path = (
    "/Users/rafaelvalle//Desktop/kaldi/src/online2bin/online2-wav-dump-features"
options = ("--config=/Users/rafaelvalle//Desktop/kaldi_online_fisher/"+
    "nnet_a_gpu_online/conf/online_nnet2_dump.conf --verbose=1"
spk2utt_rspecifier = "ark:echo utterance-id1 utterance-id1"
wav_rspecifier = "scp:echo utterance-id1 audio/clinton1_8k.wav"
feature_wspecifier = "ark,t:features/clinton1_8k.ark"

data = extract_features(wav_dump_features_path, options, spk2utt_rspecifier,
    wav_rspecifier, feature_wspecifier)
"""

# read kaldi features and transform them into numpy array
feature_path = "features/clinton1_8k.ark"
data = read_kaldi_features(feature_path)

# create generator with spliced data and iVectors
data_gen = splice(data, left_context, right_context, const_component_dim)

# compute ouputs using neural network layers. first two items in list are for
# description
predictions = [predict(i, net[3:]) for i in data_gen]