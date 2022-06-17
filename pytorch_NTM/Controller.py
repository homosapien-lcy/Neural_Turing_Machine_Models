from MemoryBank import *


class Controller(nn.Module):
    def __init__(self, model, bank_size, key_size, y_size):
        super().__init__()

        self._model = model(num_classes=key_size)
        self._key_size = key_size
        self._bank_size = bank_size
        self._y_size = y_size

        print("creating memory bank- bank size: " + str(self._bank_size) + " key size: " + str(self._key_size))

        self.memory_bank = MemoryBank(self._bank_size, self._key_size, self._y_size)

    def forward(self, x, norm_key=False):
        keys = self._model(x)

        # if norm_key, normalize key to the sphere
        if norm_key:
            keys = normalize(keys, p=2, d=1)

        prophecy = self.memory_bank.read(keys)
        return prophecy

    def memorize(self, x, y, write_rate):
        keys = self._model(x)
        self.memory_bank.write(keys, y, write_rate)

    def forget(self, forget_cut, method="rank", forget_amount=3):
        print("forgetting unused memories")
        self.memory_bank.forget_by_usage(method="rank", param=forget_amount)
        print("forgetting duplicated memories")
        self.memory_bank.forget_by_duplication(forget_cut, forget_amount)

    def L1_bank_y_reg(self, reg_rate):
        bank_y_loss = self.memory_bank.y_loss()
        return bank_y_loss * reg_rate

    def cuda(self):
        self._model.cuda()
        self.memory_bank.cuda()

    def cpu(self):
        self._model.cpu()
        self.memory_bank.cpu()
