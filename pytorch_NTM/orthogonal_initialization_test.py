import numpy as np
import time
import scipy.optimize as skopt


# square sum of cosine similarity
def sqscs(A, x):
    y = np.dot(A, x)
    return np.dot(y, y)


# function that provide the initial solution
# to the minimizer
def flat_init(vec_len):
    if vec_len == 0:
        print("length should not be 0!!!!!")
        return np.array([])

    each_element = np.sqrt(1 / vec_len)
    return np.array([each_element] * vec_len)

old_key_mat = np.load("key_bank_status.npy")[:11]

start = time.time()

target_fun = lambda x: sqscs(key_space_vectors, x)
# spherical constraint
constraint_fun = lambda x: np.dot(x, x) - 1
insert_vector = lambda mat, vec: np.concatenate([vec.reshape(1, -1),
                                                 mat], axis=0)

cons = ({'type': 'eq', 'fun': constraint_fun})

pos_bound = True
_key_size = 8

if pos_bound:
    bds = [(0, 1)] * _key_size
else:
    bds = None

new_key_box = []
# key space vectors to solve
key_space_vectors = np.array(old_key_mat)
# the new keys that minimize the absolute angle
for i in range(0, 10):
    initial_point = flat_init(_key_size)
    opt_out = skopt.minimize(fun=target_fun,
                             x0=initial_point,
                             constraints=cons,
                             bounds=bds)
    solution = opt_out['x']
    new_key_box.append(solution)
    print(solution)
    print(target_fun(solution))
    key_space_vectors = insert_vector(key_space_vectors, solution)

end = time.time()
print("total time: " + str(end - start) + " seconds")
