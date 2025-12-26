import numpy as np
from scipy.io import loadmat

def checkBursts(a=None):
    if a is None:
        a = loadmat('channel_state_2.mat')['channel_state_2']
        print('loaded from file')
    len0_lists = []
    len1_lists = []
    for i in range(len(a)):
        len0_list = []
        len1_list = []
        len0 = 0
        len1 = 0
        on_1 = bool(a[i,0])
        on_0 = ~on_1
        for j in range(len(a[i])):
            if a[i,j] == 0:
                len0+=1
                on_0 = True
                if on_1:
                    len1_list.append(len1)
                    len1 = 0
                    on_1 = False
                
            elif a[i,j] == 1:
                len1+=1
                on_1 = True
                if on_0:
                    len0_list.append(len0)
                    len0 = 0
                    on_0 = False
        len0_lists.append(len0_list)
        len1_lists.append(len1_list)

    for i in range(len(a)):
        print(f'bit 0 for PU {i}',np.mean(len0_lists[i]))
        print(f'bit 1 for PU {i}',np.mean(len1_lists[i]))

def burstyData(p0,p1,len0,num_bits,RandomState=None):
    '''This function will create a stream of bits of length
    num_bits that follows the bursty data distribution.
    Arguments:
    p0: the probability of bit 0 happening
    p1: the probability of bit 1 happening
    len0: the average length of consecutive zeros
    num_bits = total number of bits to be generated'''
    assert p0+p1 == 1
    bit_stream = np.zeros(num_bits)
    if RandomState is None:
        RandomState = np.random.RandomState()
    state = 0
    p_transition_from_0 = 1/len0
    p_transition_from_1 = p0*p_transition_from_0/p1
    print(f'Probability of getting out of 0 is {p_transition_from_0}\n Probability of getting out of 1 is {p_transition_from_1}')
    assert (0<p_transition_from_0<1 and 0<p_transition_from_1<1)
    for i in range(num_bits):
        if state == 0:
            if RandomState.rand() < p_transition_from_0:
                state = 1
                bit_stream[i] = 1
            else:
                bit_stream[i] = 0
        elif state == 1:
            if RandomState.rand() < p_transition_from_1:
                state = 0
                bit_stream[i] = 0
            else:
                bit_stream[i] = 1


    return bit_stream