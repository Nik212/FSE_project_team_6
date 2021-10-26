# ENet: https://arxiv.org/abs/1606.02147
# based off of https://github.com/bermanmaxim/Enet-PyTorch

from functools import reduce
import torch
from torch import nn

from torch.autograd import Variable


class LambdaBase(nn.Sequential):
    """
    class
    """
    def __init__(self, f_n, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = f_n

    def forward_prepare(self, input):
        """
        method
        """
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    """
    class
    """
    def forward(self, input):
        """
        method
        """
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    """
    class
    """
    def forward(self, input):
        """
        result is Variables list [Variable1, Variable2, ...]
        """
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    """
    class
    """
    def forward(self, input):
        """
        result is a Variable
        """
        return reduce(self.lambda_func, self.forward_prepare(input))


class Padding(nn.Module):
    """
     pad puts in [pad] amount of [value] over dimension [dim], starting at
     index [index] in that dimension. If pad<0, index counts from the left.
     If pad>0 index counts from the right.
     When n_input_dim is provided, inputs larger than that value will be considered batches
     where the actual dim to be padded will be dimension dim + 1.
    """
    def __init__(self, dim, pad, value, index, n_input_dim):
        super(Padding, self).__init__()
        self.value = value
        # self.index = index
        self.dim = dim
        self.pad = pad
        self.n_input_dim = n_input_dim
        if index != 0:
            raise NotImplementedError("Padding: index != 0 not implemented")

    def forward(self, input):
        """
        forward
        """
        dim = self.dim
        if self.n_input_dim != 0:
            dim += input.dim() - self.n_input_dim
        pad_size = list(input.size())
        pad_size[dim] = self.pad
        padder = Variable(input.data.new(*pad_size).fill_(self.value))

        if self.pad < 0:
            padded = torch.cat((padder, input), dim)
        else:
            padded = torch.cat((input, padder), dim)
        return padded


class Dropout(nn.Dropout):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """

    def forward(self, input):
        """
        forward
        """
        input = input * (1 - self.p)
        return super(Dropout, self).forward(input)


class Dropout2d(nn.Dropout2d):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """

    def forward(self, input):
        """
        forward
        """
        input = input * (1 - self.p)
        return super(Dropout2d, self).forward(input)


class StatefulMaxPool2d(nn.MaxPool2d):
    """
    object keeps indices and input sizes
    """

    def __init__(self, *args, **kwargs):
        """
        init
        """
        super(StatefulMaxPool2d, self).__init__(*args, **kwargs)
        self.indices = None
        self.input_size = None

    def forward(self, _x):
        """
        init
        """
        return_indices, self.return_indices = self.return_indices, True
        output, indices = super(StatefulMaxPool2d, self).forward(_x)
        self.return_indices = return_indices
        self.indices = indices
        self.input_size = _x.size()
        if return_indices:
            return output, indices
        return output


class StatefulMaxUnpool2d(nn.Module):
    """
    class
    """
    def __init__(self, pooling):
        """
        init
        """
        super(StatefulMaxUnpool2d, self).__init__()
        self.pooling = pooling
        self.unpooling = nn.MaxUnpool2d(pooling.kernel_size, pooling.stride, pooling.padding)

    def forward(self, _x):
        """
        forward
        """
        return self.unpooling.forward(_x, self.pooling.indices, self.pooling.input_size)


POOLING_0 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)
POOLING_1 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)
POOLING_2 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)


def create_enet(num_classes):
    """
    @rtype: object
    """
    enet = nn.Sequential(  # Sequential,
        LambdaReduce(lambda x, y: torch.cat((x, y), 1)),
        LambdaMap(lambda x: x,  # ConcatTable,
                  nn.Conv2d(3, 13, (3, 3), (2, 2), (1, 1), (1, 1), 1),
                  POOLING_0,),
        nn.BatchNorm2d(16, 0.001, 0.1, True),
        nn.PReLU(16),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(16, 16, (2, 2), (2, 2), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                          POOLING_1,
                          Padding(0, 48, 0, 0, 3),
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(16, 0.001, 0.1, True),
                          nn.PReLU(16),
                          nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(64, 0.001, 0.1, True),
                          Dropout2d(0.01),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(64, 32, (2, 2), (2, 2), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                          POOLING_2,
                          Padding(0, 64, 0, 0, 3),
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (1, 5), (1, 1), (0, 2), (1, 1), 1, bias=False),
                          nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(lambda x: x,  # ConcatTable,
                      nn.Sequential(  # Sequential,
                          nn.Conv2d(128, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1),
                          nn.BatchNorm2d(32, 0.001, 0.1, True),
                          nn.PReLU(32),
                          nn.Conv2d(32, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
                          nn.BatchNorm2d(128, 0.001, 0.1, True),
                          Dropout2d(0.1),
                      ),
                      nn.Sequential(  # Sequential,
                          Lambda(lambda x: x),  # Identity,
                      ),),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(
            nn.Conv2d(128, num_classes, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False)
        )
    )
    return enet


def create_enet_for_3d(type, model_path, num_3d_classes):
    """

    @rtype: object
    """
    num_classes = type[0]
    model = create_enet(num_classes)
    model.load_state_dict(torch.load(model_path))
    # remove the classifier
    _n = len(model)
    model_trainable = nn.Sequential(*(model[i] for i in range(_n - 9, _n - 1)))
    model_fixed = nn.Sequential(*(model[i] for i in range(_n - 9)))
    model_classifier = nn.Sequential(nn.Conv2d(128, num_3d_classes, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False))
    for param in model_fixed.parameters():
        param.requires_grad = False
    return model_fixed, model_trainable, model_classifier
