#add to user-defined parameters
import argparse

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='path to save log and checkpoint.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size to use.')

    parser.add_argument('--input_file_pattern', type=str, default='/data/weixin-42421001/flickr8k/train-?????-of-00004',
                        help='File pattern of sharded TFRecord file containing SequenceExample protos.')

    parser.add_argument('--inception_checkpoint_file', type=str, default='/data/weixin-42421001/flickr8k/inception_v3.ckpt',
                        help='File containing an Inception v3 checkpoint to initialize the variables')
    
    parser.add_argument('--embedding_size', type=int, default=512,
                        help='LSTM input dimensionality.')

    parser.add_argument('--num_lstm_units', type=int, default=512,
                        help='LSTM output dimensionality.')

    parser.add_argument('--lstm_dropout_keep_prob', type=float, default=0.7,
                        help='If < 1.0, the dropout keep probability applied to LSTM variables.')

    #
    parser.add_argument('--num_examples_per_epoch', type=int, default=586363,
                        help='Number of examples per epoch of training data.')

    parser.add_argument('--initial_learning_rate', type=float, default=2.0,
                        help='Learning rate for the initial phase of training.')
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.5,
                        help='Learning rate for the initial phase of training.')
    parser.add_argument('--num_epochs_per_decay', type=int, default=8,
                        help='Learning rate for the initial phase of training.')

    parser.add_argument('--train_inception_learning_rate', type=float, default=0.0005,
                        help='Learning rate when fine tuning the Inception v3 parameters.')

    parser.add_argument('--clip_gradients', type=float, default=5.0,
                        help='If not None, clip gradients to this value.')

    parser.add_argument('--max_checkpoints_to_keep', type=int, default=10,
                        help='How many model checkpoints to keep.')

    #
    parser.add_argument('--train_dir', type=str, default='/output/train',
                        help='Directory for saving and loading model checkpoints.')

    parser.add_argument('--train_inception', type=bool, default=False,
                        help='Whether to train inception submodel variables.')

    parser.add_argument('--number_of_steps', type=int, default=1000000,
                        help='Number of training steps.')

    parser.add_argument('--log_every_n_steps', type=int, default=1,
                        help='Frequency at which loss and global step are logged.')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    for x in dir(FLAGS):
        print(getattr(FLAGS, x))
