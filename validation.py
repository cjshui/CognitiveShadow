import numpy as np
import argparse

def our_sign(x):

    if x>= 0:

        return  1
    else:
        return -1


def net_out(x,w11,w12,w10,w21,w22,w20):

    x1 = x[0]
    x2 = x[1]

    O1 = our_sign(x1*w11 + x2*w12 + w10)
    O2 = our_sign(x2*w21 + x2*w22 + w20)

    O3 = our_sign(1.0*O1 + 1.0*O2 - 1.5)


    return O3


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input weight, should be w11, w12, w10, w21, w22, w20')

    parser.add_argument('--weights', nargs='+', type=float)

    args = parser.parse_args()

    weight = args.weights

    # print(weight)

    w11,w12,w10,w21,w22,w20 = weight

    X = [[0.5,0.3],[0.4,0.3],[0.3,0.35],[0.4,0.4],[0.45,0.45],[0.5,0.35],[0.55,0.45],[0.4,0.55]]
    y = [-1,-1,-1,-1,1,1,1,1]

    error = 0

    for i in range(8):
        if y[i] != net_out(X[i],w11,w12,w10,w21,w22,w20):

            error += 1


    print("The total score is {}".format(1-error/8))







