import numpy as np

N_r = 2
N_t = 2
W = 3
T = 8
N_sym = 14
N_sc = 512
T_s = T * N_sym * N_sc
F = N_r * N_t * W * (W+1)
first_term = T_s * (N_r**2 * N_t * W + N_r * N_t**2 * W**2)
second_term = F**2 * T_s
third_term = F**3
fourth_term = N_r * N_t * T_s * F
term_wise_complexities = np.asarray([first_term, second_term, third_term, fourth_term])

print("")
for i,j in enumerate(term_wise_complexities):
    print("{}th term complexity: {}".format(i+1,j))

dominant_term = np.argmax(term_wise_complexities)
print("\nMax complexity term for unconfigured WESN: {}".format(dominant_term+1))
print("Max complexity: {}".format(term_wise_complexities[dominant_term]))

hold = 1