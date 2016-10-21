import time, threading
import subprocess
import pdb
import numpy as np
import OSC
import kaldi_nnet_tools as knt

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
 

def _splice_features(filename):
    left_context = abs(clf[2]['<Context>'][0])
    right_context = abs(clf[2]['<Context>'][-1])
    const_component_dim = clf[2]['<ConstComponentDim>']

    # create generator with spliced data and iVectors
    data = knt.splice(
        knt.read_kaldi_features("features/{}.ark".format(filename)), 
        left_context, 
        right_context, 
        const_component_dim)

def _splice_features(addr, tags, data, source):
    _splice_features(data[0])

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

# define addresses
receive_address = '127.0.0.1', 31337
send_address = '127.0.0.1', 12345

# start server and client
osc_server = OSC.OSCServer(receive_address)
osc_server.addDefaultHandlers()
osc_client = OSC.OSCClient()
osc_client.connect( send_address )

# add message handlers
osc_server.addMsgHandler("/print_handlers", printing_handler)
osc_server.addMsgHandler("/extract_features", extract_features)
osc_server.addMsgHandler("/decode_loglikelihoods", decode_loglikelihoods)
osc_server.addMsgHandler("/compute_loglikelihoods", compute_loglikelihoods)
osc_server.addMsgHandler("/decode_audio", decode_audio)

print "Instantiate ANN Classifier"
# path where model info nnet-am-info and copy nnet-am-copy are saved
am_copy_path = 'models/fisher_final.mdl.nnet.txt'
am_info_path = 'models/fisher_final.mdl.info.txt'

# convert kaldi model to python
clf = knt.parseNNET(am_copy_path, am_info_path)

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
