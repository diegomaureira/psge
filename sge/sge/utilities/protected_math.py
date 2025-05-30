from math import log, exp, sqrt


def _log_(x):
    if x <= 0: return 0
    return log(x)


def _sig_(x):
    return 1.0 / (1.0 + _exp_(-x))


def protdiv(x, y):
    if y == 0:
        return 1
    return x / y


def _exp_(x):
    try:
        return exp(x)
    except ValueError:
        return 1


def _inv_(x):
    if x == 0: return 1
    return 1.0 / x


def _sqrt_(x):
    return sqrt(abs(x))

def _pow_(base, exponent):
    try:
        return base**exponent
    except:
        return 0

def getcenterdistance(x,y, imgsize):
    center = (int(imgsize[0]/2.0),
              int(imgsize[1]/2.0))
    return sqrt(pow(x-center[0],2) + pow(y-center[1],2)) / sqrt(pow(center[0],2) + pow(center[1],2))

class Infix:
    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


_div_ = Infix(protdiv)

if __name__ == '__main__':
    print(8 | _div_ | 2)
    print(9.0 | _div_ | 2)
    print(8 | _div_ | 0)
    print(8 / 0)
