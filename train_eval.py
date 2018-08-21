from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from flags import parse_args

if __name__ == '__main__':
	FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    cmd = ""
    for parm in ["output_dir", "batch_size", "input_file_pattern", "inception_checkpoint_file",
     "embedding_size", "num_lstm_units", "lstm_dropout_keep_prob", "num_examples_per_epoch", 
     "initial_learning_rate", "learning_rate_decay_factor", "num_epochs_per_decay", 
     "train_inception_learning_rate", "clip_gradients", "max_checkpoints_to_keep", 
     "train_dir", "train_inception", "number_of_steps", "log_every_n_steps"]:
        try:
            cmd += ' --{0}={1}'.format(parm, getattr(FLAGS, parm))
        except:
            pass


    for i in range(10):
       # train 1 epoch
        print('################    train    ################')
        p = os.popen('python ./train.py' + cmd)
        for l in p:
            print(l.strip())

        # eval
        print('################    eval    ################')
        p = os.popen('python ./evaluate.py' + cmd)
        for l in p:
            print(l.strip())
