import sys

sys.path.append('../Common_Code/')
from MemoryBank_Utils import *


class MemoryBank():
    def __init__(self, bank_size, key_size, y_size):

        self._bank_size = bank_size
        self._key_size = key_size
        self._y_size = y_size

        # key use randn, both + and -
        self.key_bank = torch.randn(self._bank_size, self._key_size)
        # y use rand, all positive
        self.y_bank = torch.rand(self._bank_size, self._y_size)
        # usage data for instance filtering
        self.usage = torch.ones(self._bank_size) / self._bank_size

        #self.min_cos_sim_init(pos_bound=False)
        self.bank_normalize()
        self.hard_y()

    # initialize y_bank to be either 1 or 0
    def hard_y(self):
        self.y_bank = normalize(self.y_bank, p=1)
        self.y_bank = make_one_hot(torch.argmax(self.y_bank, dim=1),
                                   C=self._y_size, device="cpu")

    # only initialize keys in the positive sphere
    def pos_sphere_init(self):
        self.key_bank = torch.rand(self._bank_size, self._key_size)

    # minimum cosine similarity initialization
    def min_cos_sim_init(self, pos_bound, rfk=revise_first_key):
        keep_list = [0]
        remove_list = range(1, self._bank_size)

        # make the first key to be (1, 0, 0, ...) basis
        if rfk:
            self.key_bank[0] = torch.zeros(self._key_size)
            self.key_bank[0, 0] = 1

        new_key_box = self.abs_ms_key_gen(keep_list, remove_list,
                                          pos_bound=pos_bound, print_new=True)

        # replacing old keys with orthogonal keys
        for i in range(0, len(remove_list)):
            key_to_reinit = remove_list[i]
            self.key_bank[key_to_reinit] = new_key_box[i]

    def cuda(self):
        self.key_bank = self.key_bank.cuda()
        self.y_bank = self.y_bank.cuda()
        self.usage = self.usage.cuda()

    def cpu(self):
        self.key_bank = self.key_bank.cpu()
        self.y_bank = self.y_bank.cpu()
        self.usage = self.usage.cpu()

    def cosine_sim(self, keys):
        # transpose, return size [key_size, batch_size]
        t_keys = torch.t(keys)
        # return size [batch_size, bank_size]
        return torch.t(torch.matmul(self.key_bank, t_keys))

    def cosine_sim_weight(self, keys, normalize_fun=softmax_normalize):
        cosine_sim = self.cosine_sim(keys)
        return normalize_fun(cosine_sim)

    def normalize_usage(self):
        self.usage = normalize_vec(self.usage)

    def update_usage(self, weight, momentum=0.99):
        # average weight of size [bank_size]
        average_weight = torch.mean(weight, dim=0)
        self.usage = momentum * self.usage.data + (1 - momentum) * average_weight.data
        self.normalize_usage()

    def bank_normalize(self, on_key=True, on_y=True):
        if on_key:
            # key normalize with 2 norm (sq sum=1)
            self.key_bank = normalize(mat=self.key_bank, p=2)
        if on_y:
            # y normalize with 2 norm (sq sum=1)
            self.y_bank = normalize(mat=self.y_bank, p=2)

    # generate new keys by random
    def random_key_gen(self, keep, remove):
        number_of_keys = len(remove)
        new_key_box = []
        for i in range(0, number_of_keys):
            new_key = torch.randn(self._key_size)
            new_key_box.append(new_key)

        return new_key_box

    # generate new keys by random in the positive sphere
    def pos_random_key_gen(self, keep, remove):
        number_of_keys = len(remove)
        new_key_box = []
        for i in range(0, number_of_keys):
            new_key = torch.rand(self._key_size)
            new_key_box.append(new_key)

        return new_key_box

    # generate new keys with minimum similarity principle
    def ms_key_gen(self, keep, remove):
        number_of_keys = len(remove)

        # collect the keep_set vectors
        old_key_box = []
        for i in range(0, len(keep)):
            key_to_keep = keep[i]
            old_key_box.append(self.key_bank[key_to_keep].view(1, -1))

        old_key_mat = torch.cat(old_key_box, dim=0)
        # result from Lagrangian multiplier
        sigma_a_ik = torch.sum(old_key_mat, dim=0)
        # sign should be negative for positive definite hessian
        sign = -1
        # the argmin from Lagrangian
        argmin_keyseed = normalize_vec(sigma_a_ik, p=2) * sign

        new_key_box = []
        # the new keys that solve the Lagrangian
        for i in range(0, number_of_keys):
            offset_vec = normalize_vec(torch.randn(self._key_size), p=2)
            new_key = argmin_keyseed + offset_vec
            new_key_box.append(new_key)

        return new_key_box

    # generate new keys with minimum similarity principle (absolute)
    # try to find results in the orthogonal space
    def abs_ms_key_gen(self, keep, remove, pos_bound=False, print_new=False):
        number_of_keys = len(remove)

        # collect the keep_set vectors
        old_key_box = []
        for i in range(0, len(keep)):
            key_to_keep = keep[i]
            old_key_box.append(self.key_bank[key_to_keep].view(1, -1))

        old_key_mat = torch.cat(old_key_box, dim=0)

        # square sum of cosine similarity
        def sqscs(A, x):
            y = np.dot(A, x)
            return np.dot(y, y)

        # functions that provide the initial solution
        # to the minimizer
        def flat_init(vec_len):
            if vec_len == 0:
                print("length should not be 0!!!!!")
                return np.array([])

            each_element = np.sqrt(1 / vec_len)
            return np.array([each_element] * vec_len)

        # randn is harder to stuch at saddle point than flat
        def randn_init(vec_len, eps=eps):
            if vec_len == 0:
                print("length should not be 0!!!!!")
                return np.array([])

            vec = np.random.randn(vec_len)
            return (vec / (np.dot(vec, vec) + eps))

        target_fun = lambda x: sqscs(key_space_vectors, x)
        # spherical constraint
        constraint_fun = lambda x: np.dot(x, x) - 1
        # function for insert new solution into the key space
        insert_vector = lambda mat, vec: np.concatenate([vec.reshape(1, -1),
                                                         mat], axis=0)

        cons = ({'type': 'eq', 'fun': constraint_fun})

        # if positive sphere, bound by 0 -> 1
        if pos_bound:
            bds = [(0, 1)] * self._key_size
        else:
            bds = None

        new_key_box = []
        # key space vectors to solve
        key_space_vectors = np.array(old_key_mat)
        # the new keys that minimize the absolute angle
        for i in range(0, number_of_keys):
            # calculate solution
            initial_point = randn_init(self._key_size)
            opt_out = skopt.minimize(fun=target_fun,
                                     x0=initial_point,
                                     constraints=cons,
                                     bounds=bds)
            solution = opt_out['x']
            new_key_box.append(torch.tensor(solution))

            # if require print, print
            if print_new:
                print("reinitiating point: ")
                print(solution)
                print(target_fun(solution))

            # insert new solution into the key space
            key_space_vectors = insert_vector(key_space_vectors, solution)

        return new_key_box

    def read(self, keys, usage_threshold=-1, eps=eps):
        # return size [batch_size, bank_size]
        weight = self.cosine_sim_weight(keys)

        # square the y to make sq sum=1
        selected_y_bank = self.y_bank.pow(2)

        # remove instance with usage < usage_threshold
        # during prediction
        if usage_threshold > 0.0:
            # get the cut from set threshold and max in usage
            cut = min(usage_threshold, torch.max(self.usage).item() - eps)

            # make the selected bank
            above_threshold = (F.threshold(self.usage, cut, 0) > 0).float()
            instance_selector = torch.stack([above_threshold[:]] * self._y_size, dim=1)
            selected_y_bank = selected_y_bank * instance_selector.clone().data

        # return size [batch_size, y_size]
        pred = torch.matmul(weight, selected_y_bank)
        # normalize for threshold case
        pred = normalize(pred, p=1, d=1)

        # update the usage vector
        self.update_usage(weight)

        # ys are saved in 2 norm and output in 1 norm
        # so that we can use l1_loss
        return pred

    def write(self, keys, y, write_rate):
        # return size [batch_size, y_size]
        y_onehot = make_one_hot(y, self._y_size).float()

        # return size [bank_size, batch_size]
        t_weight = torch.t(self.cosine_sim_weight(keys))

        # update keys
        # return size [bank_size, key_size]
        key_update = torch.matmul(t_weight, keys)
        # use standardize, softmax and normalize to
        # to increase difference between updates
        key_update = standardize(key_update)
        key_update = F.softmax(key_update, dim=1)
        key_update = normalize(key_update, p=2)
        self.key_bank += key_update.data * write_rate

        # update ys
        # return size [bank_size, y_size]
        y_update = torch.matmul(t_weight, y_onehot)
        y_update = standardize(y_update)
        y_update = F.softmax(y_update, dim=1)
        y_update = normalize(y_update, p=2)
        self.y_bank += y_update.data * write_rate

        self.bank_normalize()

    # helper method for the two forget methods
    # get new key and forget
    def forget_helper(self, keep_list, remove_list,
                      reinit_method=abs_ms_key_gen):
        # get new key
        new_key_box = reinit_method(self, keep_list, remove_list)
        print("new key box")
        print(new_key_box)
        # reinitiating parameters from the remove list
        for i in range(0, len(remove_list)):
            key_to_reinit = remove_list[i]
            print("reinitiating: " + str(key_to_reinit))
            # reset usage
            self.usage[key_to_reinit] = 0.0 + eps
            # replace keys -
            self.key_bank[key_to_reinit] = new_key_box[i]
            # y use rand, all positive
            self.y_bank[key_to_reinit] = nan_to_zero(torch.rand(self._y_size),
                                                     regenerator=torch.rand)

    # forget by usage
    # two possible mode: "cutoff" is usage < certain percentage, where param is the cutoff
    #                    "rank" is below certain rank, where param is the threshold rank
    # has too be run before running forget_by_duplication!!!
    def forget_by_usage(self, method="cutoff", param=1e-2):
        if method == "cutoff":
            cutoff = param
            # make remove list
            remove_arr = np.argwhere(np.array(self.usage) < cutoff).reshape(-1)
            remove_list = list(remove_arr)

            # make keep list
            whole_set = set(range(0, self._bank_size))
            remove_set = set(remove_list)
            keep_set = whole_set - remove_set
            keep_list = list(keep_set)
        elif method == "rank":
            num_forget = param
            # make remove list
            usage_rank = np.argsort(np.array(self.usage))
            remove_arr = usage_rank[:num_forget]
            remove_list = list(remove_arr)

            # make keep list
            whole_set = set(range(0, self._bank_size))
            remove_set = set(remove_list)
            keep_set = whole_set - remove_set
            keep_list = list(keep_set)
        else:
            print("What the hell is the method of your choice!!! Method has to be either cutoff or rank.")
            return

        self.forget_helper(keep_list, remove_list)

    # forget the replicated keys in the key bank
    def forget_by_duplication(self, forget_cut, forget_amount):
        # key_bank cosine similarity, don't need to worry about
        # absolute value, since key exist in a sphere and two sides
        # are different instances
        key_bank_cs = torch.matmul(self.key_bank, torch.t(self.key_bank))
        print("bank cosine similarity")
        print(key_bank_cs)
        # remove the diagonal element
        key_bank_cs = key_bank_cs.cpu() - torch.diag(torch.ones(self._bank_size))
        # find duplicated keys
        duplicated = np.argwhere(np.array(key_bank_cs) > forget_cut)
        print("duplicated pairs:")
        print(duplicated)

        ori_set = set([])
        dup_set = set([])
        nan_set = set([])

        for i in range(0, self._bank_size):
            if check_mat_nan(self.y_bank[i]):
                nan_set.add(i)

        for i in range(0, len(duplicated)):
            r = duplicated[i, 0]
            c = duplicated[i, 1]
            # adding rule, r only added to ori if not in duplicate
            # set will handle the duplicates
            if r not in dup_set:
                ori_set.add(r)

            # adding rule, c only added to duplicate if not in original
            # set will handle the duplicates
            if c not in ori_set:
                dup_set.add(c)

        print("original keys")
        print(ori_set)
        print("duplicated keys")
        print(dup_set)
        print("nan set")
        print(nan_set)
        print(self.usage)

        # set of all numbers
        whole_set = set(range(0, self._bank_size))
        keep_set = whole_set - dup_set - nan_set
        remove_set = dup_set - nan_set

        if (len(remove_set) + len(nan_set)) != 0:
            keep_list = list(keep_set)
            whole_remove_list = list(remove_set)

            # only forget a few at a time
            remove_list = whole_remove_list[:forget_amount]
            nan_list = list(nan_set)

            # extract usage for redistribution
            usage_to_redist = self.usage.clone()[remove_list].cpu()
            cs_for_redist = key_bank_cs.clone()[remove_list]

            # also forget all nans
            remove_list.extend(nan_list)

            # move to cpu for loop operations
            self.cpu()

            # calculate the distribution and update usage
            cs_for_redist[:, remove_list] = -9999
            redist_weight = softmax_normalize(cs_for_redist)
            usage_share = torch.matmul(usage_to_redist, redist_weight)
            self.usage += usage_share.data

            # forget
            self.forget_helper(keep_list, remove_list)

            # check usage calculation
            if abs(torch.sum(self.usage) - 1) > 0.1:
                print("usage sum has large error!!!!!!!!!!!!!!")
                print("value off: ")
                print(str(torch.sum(self.usage) - 1))
                print("redist share: ")
                print(usage_share)
                print("usage value: ")
                print(self.usage)
                print("during forgetting")

            # move back to gpu
            self.cuda()
            self.bank_normalize()
            self.normalize_usage()

            print("updated key bank")
            print(self.key_bank.pow(2))
            print("updated y bank")
            print(self.y_bank.pow(2))

    def y_loss(self):
        # l1 loss
        return torch.sum(torch.abs(self.y_bank))

'''
def test_1():
    test_bank = MemoryBank(10, 5, 4)
    test_bank.y_bank[2, 1] = np.nan
    test_bank.y_bank[3, 1] = np.nan
    print(test_bank.y_bank)
    test_bank.usage[0] = 5
    test_bank.forget_by_usage(method="cutoff", param=0.2)
    test_bank.forget_by_duplication(0.05, 3)
    print(test_bank.y_bank)


def test_2():
    test_bank = MemoryBank(10, 5, 4)
    print(test_bank.key_bank)
    test_bank.key_bank[-1] = torch.tensor([0, 0, 0, 1, 0])
    print(test_bank.key_bank)

test_2()
'''
