import numpy as np
import matplotlib.pyplot as plt

def RNN_CN(B_rec, d_y, d_h, d_x, t, epsilon=1, rho_y=1 , rho_h=1 ,B_in=1 , B_out=1 , B_X=1, ):
    const = (rho_h*rho_y*B_in*B_out*B_X/epsilon)**2
    upperBound = const*(
        9*((rho_h**t*B_rec**t - 1)/(rho_h*B_rec-1))**2 * np.log(2*d_y*d_h) +
        9* t**2 * rho_h**2 * B_rec**2 * ((rho_h**(2*t)*B_rec**(2*t) - 1)/(rho_h**2*B_rec**2-1)) * np.log(2*d_h*d_x) + 
        9* (t-1)**2 / ((rho_h*B_rec-1)**2) * (
            (t-1)*(rho_h*B_rec)**(2*t) - 
            2*rho_h**(t+1)*B_rec**(t+1) * (((rho_h*B_rec)**(t-1) - 1)/((rho_h*B_rec) - 1)) +
            rho_h**2 * B_rec**2 * (((rho_h*B_rec)**(2*t-1) - 1)/((rho_h*B_rec)-1))
        ) * np.log(2*d_h*d_h)
    )
    return upperBound

def ESN_CN(B_rec, d_y, d_h, d_x, t, epsilon=1, rho_y=1 , rho_h=1 ,B_in=1 , B_out=1 , B_X=1, ):
    const = (rho_h*rho_y*B_in*B_out*B_X/epsilon)**2
    upperBound = const * d_y * ((rho_h**t*B_rec**t - 1)/(rho_h*B_rec-1))**2 * np.log(2*d_h*d_y)
    return upperBound


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
    plt.title("Recurrent Weight matrix L2-Norm (for DRQN) ")
    plt.show()

    d_x = n_channel + 1
    d_h = 64
    d_y = n_channel*2
    B_rec_rnn = 0.7

    B_rec_esn = 0.7
    t_rnn = 5
    t_esn = 5

    print('RNN:',RNN_CN(B_rec_rnn,d_y,d_h,d_x,t_rnn))
    print('ESN:',ESN_CN(B_rec_esn,d_y,d_h,d_x,t_esn))

    