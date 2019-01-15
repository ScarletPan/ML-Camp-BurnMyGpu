def get_elmo_model():
    from allennlp.modules.elmo import Elmo
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/" \
                   "elmo/2x4096_512_2048cnn_2xhighway/" \
                   "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/" \
                  "elmo/2x4096_512_2048cnn_2xhighway/" \
                  "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    return elmo