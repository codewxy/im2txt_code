from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

if __name__ == '__main__':
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)


    for i in range(10):
       # train 1 epoch
        print('################    train    ################')
        p = os.popen('python ./train.py' )
        for l in p:
            print(l.strip())

        # eval
        print('################    eval    ################')
        p = os.popen('python ./evaluate.py' )
        for l in p:
            print(l.strip())
