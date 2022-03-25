import numpy as np
import matplotlib.pyplot as plt
import argparse
from svm import SVCTrainer


def parse_argumetns():

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default='rbf', choices=['rbf', 'linear'],
                        help='type of kernel function')
    parser.add_argument('--C', type=float, default=1,
                        help='regularization paramter')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='coeficient for rbf. gamma * norm(xi - xj)**2')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='tolerance to check KKT condition')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed of random numbers')

    return parser.parse_args()


def main(args):

    if args.seed > 0:
        np.random.seed(args.seed)

    # create training data
    X = np.random.randn(100, 2)
    y = np.zeros(100)
    # positive data
    X[:50] += 2
    y[:50] = 1
    # negative data
    y[50:] = -1

    XY = np.concatenate([X, y[:, np.newaxis]], axis=1)
    np.random.shuffle(XY)
    X, y = XY[:, :2], XY[:, 2]

    # svm training
    print('svm training ..')
    trainer = SVCTrainer(kernel=args.kernel,
                         C=args.C, gamma=args.gamma, tol=args.tol)
    model = trainer.train(X, y)
    support_vectors = model.get_support_vecotrs()
    print(f'the number of support vectors is {len(support_vectors)}')

    # meas accuracy
    X_pos = X[y == 1]
    X_neg = X[y == -1]
    score_pos = np.zeros(X_pos.shape[0])
    score_neg = np.zeros(X_neg.shape[0])
    for i in range(X_pos.shape[0]):
        score_pos[i] = model.predict(X_pos[i])
    for i in range(X_neg.shape[0]):
        score_neg[i] = model.predict(X_neg[i])
    acc = (np.sum(score_pos >= 0) + np.sum(score_neg < 0)) / X.shape[0]
    print(f'accuray is {acc:.3f}')

    # plot training data
    plt.figure(1)
    plt.scatter(X_pos[:, 0], X_pos[:, 1])
    plt.scatter(X_neg[:, 0], X_neg[:, 1])
    plt.xlim([-4, 6])
    plt.ylim([-4, 6])
    plt.legend(['positive', 'negative'])
    plt.grid(True)
    plt.savefig('training_data.jpg')

    # plot contour
    print('create contour ..')
    grid = np.arange(-4, 6, 0.05)
    Z1, Z2 = np.meshgrid(grid, grid)
    score_map = np.zeros_like(Z1)
    for i in range(Z1.shape[0]):
        for j in range(Z1.shape[1]):
            score_map[i, j] = model.predict(np.array([Z1[i, j], Z2[i, j]]))

    if args.kernel.lower() == 'rbf':
        ttl = f'{args.kernel.upper()} (C={args.C}, gamma={args.gamma})'
    elif args.kernel.lower() == 'linear':
        ttl = f'{args.kernel.upper()} (C={args.C})'

    plt.figure(2)
    plt.title(ttl)
    plt.contourf(Z1, Z2, score_map, alpha=0.5, cmap='jet')
    plt.colorbar()
    plt.scatter(X_pos[:, 0], X_pos[:, 1])
    plt.scatter(X_neg[:, 0], X_neg[:, 1])
    plt.xlim([-4, 6])
    plt.ylim([-4, 6])
    plt.grid(True)
    plt.savefig('training_result.jpg')
    # plt.show()


if __name__ == '__main__':
    args = parse_argumetns()
    main(args)
