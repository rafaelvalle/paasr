import time, threading
import subprocess
import cPickle as pkl
import numpy as np
import OSC
import deepdish
import theano
from theano import tensor as T
import lasagne
import neural_networks
import kaldi_nnet_tools as knt
from params import nnet_params, standard_scaler_path
import pdb

##############
#   METHODS  #
##############
def printing_handler(addr, tags, data, source):
    print "---"
    print "received new osc msg from %s" % OSC.getUrlStr(source)
    print "with addr : %s" % addr
    print "typetags %s" % tags
    print "data %s" % data
    print "---"


def _extract_features(filename):
    subprocess.call("./kaldi_extract_features.sh {}".format(filename), shell=True)


def extract_features(addr, tags, data, source):
    _extract_features(data[0])
    print "done extract_features"


def _decode_loglikelihoods(filename):
    subprocess.call("./kaldi_decode_loglikelihoods.sh {}".format(filename), shell=True)


def decode_loglikelihoods(addr, tags, data, source):
    _decode_loglikelihoods(data[0])
    print "done decode_loglikelihoods"


def _compute_loglikelihoods(filename):
    global clf
    left_context = abs(clf[2]['<Context>'][0])
    right_context = abs(clf[2]['<Context>'][-1])
    const_component_dim = clf[2]['<ConstComponentDim>']

    # create generator with spliced data and iVectors
    data = knt.splice(
        knt.read_kaldi_features("features/{}.ark".format(filename)),
        left_context,
        right_context,
        const_component_dim)

    output = [knt.forward(i, clf[3:], verbose=False) for i in data]
    output = np.clip(output, 1.0e-20, np.inf)
    output = np.log(output)
    knt.save_kaldi_loglikelihoods(
        output,
        'log_likelihoods/{}.ark'.format(filename))
    print "done compute_loglikelihoods"


def compute_loglikelihoods(addr, tags, data, source):
    _compute_loglikelihoods(data[0])


def decode_audio(addr, tags, data, source):
    global clf
    _extract_features(data[0])
    _compute_loglikelihoods(data[0])
    _decode_loglikelihoods(data[0])


def kws(addr, tags, data, source):
    """Computes class probability using a list of MFCC sent by the real-time
    feature extractor
    params
    ------
    data : array <float>
        List of floats sent by the feature extractor
    """
    data = np.array(data).reshape((N_ROWS, N_COLS))
    data = np.append(data, data[:, -3:]).reshape((1, 13, 101))
    print(pred_fn(data))


def dnn(dnn_filepath, nnet_params):
    """Loads a lasagne saved model and instantiates a function for prediction
    """
    ss = pkl.load(open(standard_scaler_path, 'rb'))
    nnet_params['offset'] = ss.mean_
    nnet_params['scale'] = ss.scale_

    network = neural_networks.build_general_network(
        (None, 13, 101),  # last is target
        nnet_params['n_layers'],
        nnet_params['widths'],
        nnet_params['non_linearities'],
        nnet_params['offset'],
        nnet_params['scale'],
        drop_out=False)

    # load best network model so far
    parameters = deepdish.io.load(dnn_filepath)

    # set network weights
    for i in xrange(len(parameters)):
        parameters[i] = parameters[i].astype('float32')

    lasagne.layers.set_all_param_values(network, parameters)

    # set up prediction function
    input_var = T.tensor3()
    prediction = lasagne.layers.get_output(
        network, input_var, deterministic=True)
    return theano.function([input_var], prediction)


# define addresses
receive_address = '127.0.0.1', 31337
send_address = '127.0.0.1', 12345

# feature
N_ROWS = 13
N_COLS = 98

dnn_filepath = "/Users/rafaelvalle/Desktop/paasr/models/kws_model.h5"
pred_fn = dnn(dnn_filepath, nnet_params)

# start server and client
osc_server = OSC.OSCServer(receive_address)
osc_server.addDefaultHandlers()
osc_client = OSC.OSCClient()
osc_client.connect(send_address)

# add message handlers
osc_server.addMsgHandler("/print_handlers", printing_handler)
osc_server.addMsgHandler("/extract_features", extract_features)
osc_server.addMsgHandler("/decode_loglikelihoods", decode_loglikelihoods)
osc_server.addMsgHandler("/compute_loglikelihoods", compute_loglikelihoods)
osc_server.addMsgHandler("/decode_audio", decode_audio)
osc_server.addMsgHandler("/kws", kws)


"""
print "Instantiate ANN Classifier"
# path where model info nnet-am-info and copy nnet-am-copy are saved
am_copy_path = 'models/fisher_final.mdl.nnet.txt'
am_info_path = 'models/fisher_final.mdl.info.txt'

# convert kaldi model to python
clf = knt.parseNNET(am_copy_path, am_info_path)
"""
# just checking which handlers we have added
print "Registered Callback-functions are :"
for addr in sorted(osc_server.getOSCAddressSpace()):
    print addr

# Start OSCServer
print "\nStarting OSCServer. Use ctrl-C to quit."
st = threading.Thread( target = osc_server.serve_forever )
st.start()


try :
    while 1 :
        time.sleep(5)
except KeyboardInterrupt :
    print "\nClosing OSCServer."
    osc_server.close()
    print "Waiting for Server-thread to finish"
    st.join() ##!!!
    print "Done!"
    import cPickle as pkl
