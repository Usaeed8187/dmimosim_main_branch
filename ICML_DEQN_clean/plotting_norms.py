import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    
    for random_seed in range(20):
        n_pu = 4
        n_su = 1
        n_channel = 3
        n_layers = 1
        model_name = 'basicDRQN'
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        WeightNormsFolder = file_folder+'norms/'
        model_name = 'basicDRQN'
        print('random_seed = ', random_seed)
        # for name in [('in_2'),('rec_2'),('out_2'),('in_F'),('rec_F'),('out_F')]:
        for name in [('rec_2')]:
            W_norms = np.load(WeightNormsFolder + 'norm_'+ name + '_' + model_name + '_seed%d.npy' % random_seed)
            for su_indx in range(len(W_norms)):
                if 0 in W_norms[su_indx]:
                    W_norms[su_indx,W_norms[su_indx]==0] = W_norms[su_indx,np.roll(W_norms[su_indx]==0,-1)]

                print(f'The matrix W_{name[:-1]} has norm-{name[-1]} of', W_norms[su_indx,-1])
    plt.plot(W_norms[su_indx,:])
    plt.xlabel('Decision Step')
    plt.ylabel('L2-norm')
    plt.title("Recurrent Weight matrix L2-Norm (for Basic Vanilla RNN) ")
    plt.grid()
    plt.show()