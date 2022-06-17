import sys
from torch.nn.parameter import Parameter

sys.path.append('../Common_Code/')
from MemoryBank_Utils import *


class DiffMemoryBank(nn.Module):
    def __init__(self, bank_size, key_size, y_size):
        super().__init__()

        self._bank_size = bank_size
        self._key_size = key_size
        self._y_size = y_size

        # key use randn, both + and -
        self.key_bank = Parameter(torch.randn(self._bank_size, self._key_size))
        # y use rand, all positive
        self.y_bank = Parameter(torch.rand(self._bank_size, self._y_size))
        # usage data for instance filtering
        self.usage = torch.ones(self._bank_size) / self._bank_size

        self.min_cos_sim_init(pos_bound=False)
        self.bank_normalize()

    def grad_on_key_bank(self):
        self.key_bank.requires_grad = True

    def grad_off_key_bank(self):
        self.key_bank.requires_grad = False

    def grad_on_y_bank(self):
        self.y_bank.requires_grad = True

    def grad_off_y_bank(self):
        self.y_bank.requires_grad = False

    def grad_on_all(self):
        self.grad_on_key_bank()
        self.grad_on_y_bank()

    def grad_off_all(self):
        self.grad_off_key_bank()
        self.grad_off_y_bank()

    def cuda(self):
        self.key_bank = Parameter(self.key_bank.cuda())
        self.y_bank = Parameter(self.y_bank.cuda())
        self.usage = self.usage.cuda()

    def cpu(self):
        self.key_bank = Parameter(self.key_bank.cpu())
        self.y_bank = Parameter(self.y_bank.cpu())
        self.usage = self.usage.cpu()

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
            self.key_bank = Parameter(normalize(mat=self.key_bank, p=2))
        if on_y:
            # y normalize with 2 norm (sq sum=1)
            self.y_bank = Parameter(normalize(mat=self.y_bank, p=2))

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
        key_space_vectors = np.array(old_key_mat.data)
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

    def cosine_sim(self, keys):
        # transpose, return size [key_size, batch_size]
        t_keys = torch.t(keys)
        # return size [batch_size, bank_size]
        return torch.t(torch.matmul(self.key_bank, t_keys))

    def cosine_sim_weight(self, keys, normalize_fun=softmax_normalize):
        cosine_sim = self.cosine_sim(keys)
        return normalize_fun(cosine_sim)

    def forward(self, x, usage_threshold=-1):
        return self.read(keys=x, usage_threshold=usage_threshold)

    def read(self, keys, usage_threshold, eps=eps):
        # return size [batch_size, bank_size]
        weight = self.cosine_sim_weight(keys)

        # square the y to make sq sum=1
        selected_y_bank = self.y_bank.pow(2)

        # remove instance with usage < usage_threshold
        # during prediction
        # may not well behave during backprop, but can
        # be kept for inference
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

    # bndw -> bank normalization during writing
    def write(self, keys, y, write_rate, bndw):
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

        if bndw:
            self.bank_normalize()

    def y_loss(self):
        # l1 loss
        return torch.sum(torch.abs(self.y_bank))


def test_bank_gen(bank_size=10):
    return DiffMemoryBank(bank_size, 5, 2)


def test_1():
    DMB = test_bank_gen()
    print("key bank")
    print(DMB.key_bank.pow(2))
    print("y bank")
    print(DMB.y_bank.pow(2))


def test_2():
    DMB = test_bank_gen()
    print("key bank grad")
    print(DMB.key_bank.requires_grad)
    print("y bank grad")
    print(DMB.y_bank.requires_grad)


def test_3():
    DMB = test_bank_gen()
    print("params")
    for p in DMB.parameters():
        print(p)
    print("named params")
    for n, p in DMB.named_parameters():
        print(n)
        print(p)
    print(DMB.y_bank.requires_grad)
    print("param dict")
    print(DMB.state_dict())


def test_4():
    DMB = test_bank_gen()
    key = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]).float()
    print("read")
    print(DMB.read(key))
    print("forward")
    print(DMB.forward(key))


def test_5():
    DMB = test_bank_gen()
    DMB.key_bank[0, 0] += 9999
    DMB.key_bank[5, 2] += 9999
    key = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]).float()
    print("key bank grad")
    print(DMB.key_bank)
    print("y bank grad")
    print(DMB.y_bank.pow(2))
    print("read")
    print(DMB.read(key))


# normalize bank frequently seems not good for training?
def test_6():
    DMB = test_bank_gen()
    key = torch.randn(10, 5)
    y_ = torch.randn(10, 2)

    '''
    DMB.grad_off_all()
    DMB.grad_on_key_bank()
    '''

    # train and optimization
    # print both y and y_ see whether
    # getting close
    criterion = nn.MSELoss()
    import torch.optim as optim
    optimizer = optim.SGD(requires_grad_filter(DMB.parameters()), lr=0.02, momentum=0.9)
    for i in range(0, 1000):
        y = DMB(key)
        loss = criterion(y, y_)

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(DMB.parameters(), 1)
        optimizer.step()

        '''
        print("label:")
        print(y_)
        print("pred:")
        print(y)
        print("error")
        print(loss.item())
        '''

        #DMB.bank_normalize()

        if i == 0:
            init_e = loss.item()
            init_y_bank = DMB.y_bank.clone()
            init_key_bank = DMB.key_bank.clone()
        elif i == 999:
            fin_e = loss.item()
            fin_y_bank = DMB.y_bank.clone()
            fin_key_bank = DMB.key_bank.clone()

    print("initial error:")
    print(init_e)
    print("final error:")
    print(fin_e)

    print("initial y bank:")
    print(init_y_bank)
    print("final y bank:")
    print(fin_y_bank)

    print("initial key bank:")
    print(init_key_bank)
    print("final key bank:")
    print(fin_key_bank)


def test_7():
    DMB = test_bank_gen(bank_size=5)
    print(DMB.usage)
    DMB.update_usage(torch.tensor([[10, 0, 0, 0, 0], [0, 0, 0, 0, 20]]).float())
    print(DMB.usage)

'''
test_7()
'''
