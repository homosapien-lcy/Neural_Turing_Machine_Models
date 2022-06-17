from DiffMemoryBank import *


class DiffController(nn.Module):
    def __init__(self, model, bank_size, key_size, y_size):
        super().__init__()

        self._model = model(num_classes=key_size)
        self._key_size = key_size
        self._bank_size = bank_size
        self._y_size = y_size

        print("creating memory bank- bank size: " + str(self._bank_size) + " key size: " + str(self._key_size))

        self.memory_bank = DiffMemoryBank(self._bank_size, self._key_size, self._y_size)

    def forward(self, x, norm_key=False):
        keys = self._model(x)

        # if norm_key, normalize key to the sphere
        if norm_key:
            keys = normalize(keys, p=2, d=1)

        prophecy = self.memory_bank(keys)
        return prophecy

    def grad_on_kernel(self):
        self._model.grad_on()

    def grad_off_kernel(self):
        self._model.grad_off()

    def grad_on_banks(self):
        self.memory_bank.grad_on_all()

    def grad_off_banks(self):
        self.memory_bank.grad_off_all()

    def grad_on_key_bank(self):
        self.memory_bank.grad_on_key_bank()

    def grad_off_key_bank(self):
        self.memory_bank.grad_off_key_bank()

    def grad_on_y_bank(self):
        self.memory_bank.grad_on_y_bank()

    def grad_off_y_bank(self):
        self.memory_bank.grad_off_y_bank()

    # memorize and forget only placed to fit the interface
    def memorize(self, x, y, write_rate, bndw=False):
        #keys = self._model(x)
        #self.memory_bank.write(keys, y, write_rate, bndw)
        pass

    def forget(self, forget_cut, method="rank", forget_amount=3):
        pass

    def memory_bank_normalize(self):
        self.memory_bank.bank_normalize()

    def L1_bank_y_reg(self, reg_rate):
        bank_y_loss = self.memory_bank.y_loss()
        return bank_y_loss * reg_rate

    def cuda(self):
        self._model.cuda()
        self.memory_bank.cuda()

    def cpu(self):
        self._model.cpu()
        self.memory_bank.cpu()


def test_controller_gen():
    return DiffController(PreActResNet18, 10, 5, 2)


def test_1():
    C = test_controller_gen()
    print(len(list(C.parameters())))
    C.grad_off_kernel()
    print(len(list(requires_grad_filter(C.parameters()))))
    C.grad_off_key_bank()
    print(len(list(requires_grad_filter(C.parameters()))))
    C.grad_off_y_bank()
    print(len(list(requires_grad_filter(C.parameters()))))
    C.grad_on_key_bank()
    print(len(list(requires_grad_filter(C.parameters()))))
    C.grad_on_kernel()
    print(len(list(requires_grad_filter(C.parameters()))))


def test_2():
    C = test_controller_gen()
    print(C.memory_bank.y_bank)
    print(C.L1_bank_y_reg(1))
    print(C._model.fc1.weight)


def test_3():
    C = test_controller_gen()
    C.cuda()

    '''
    C.grad_off_banks()
    C.grad_off_kernel()
    C.grad_on_key_bank()
    '''

    x = torch.randn(100, 3, 16, 16).cuda()
    y_ = torch.randn(100, 2).cuda()
    criterion = nn.MSELoss()
    import torch.optim as optim
    optimizer = optim.SGD(requires_grad_filter(C.parameters()), lr=10, momentum=0.9)
    for i in range(0, 100):
        y = C(x)
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

        if i == 0:
            init_e = loss.item()
            init_y_bank = C.memory_bank.y_bank.clone()
            init_key_bank = C.memory_bank.key_bank.clone()
            init_kernel = C._model.fc1.weight.clone()
        elif i == 99:
            fin_e = loss.item()
            fin_y_bank = C.memory_bank.y_bank.clone()
            fin_key_bank = C.memory_bank.key_bank.clone()
            fin_kernel = C._model.fc1.weight.clone()

        C.memory_bank_normalize()

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

    print("initial fc:")
    print(init_kernel)
    print(init_kernel.size())
    print("final fc:")
    print(fin_kernel)
    print(fin_kernel.size())


'''
test_3()
'''
