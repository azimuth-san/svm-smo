import numpy as np


class SVCTrainer:
    """Traning class for support vector classification. """

    def __init__(self, kernel='rbf', C=1, gamma=0.1, solver='smo', tol=1e-3):

        if solver.lower() == 'smo':
            self.solver = SMO(C=C, tol=tol)

        self.kernel = kernel.lower()
        self.gamma = gamma

    def train(self, X, y):
        model = SVCModel(self.kernel, self.gamma)
        return self.solver.solve(model, X, y)


class SVCModel:
    """Binary support vector classification class. """

    def __init__(self, kernel='rbf', gamma=1):
        self.gamma = gamma
        self.kernel = kernel
        if kernel == 'rbf':
            self.kernel_func = self._kernel_rbf
        elif kernel == 'linear':
            self.kernel_func = self._kernel_linear

        self.X = None
        self.y = None
        self.alpha = None
        self.b = None

    def preset(self, X, y):
        self.X = X
        self.y = y
        self.alpha = np.zeros(X.shape[0])
        self.b = 0.0

    def predict(self, input):
        # output = 0
        # for i in range(self.X.shape[0]):
        #     output = output + self.y[i] * self.alpha[i] \
        #             * self.kernel_func(self.X[i], input)
        output = np.sum(self.y * self.alpha * self.kernel_func(self.X, input))
        output += self.b
        return output

    def get_support_vecotrs(self):
        return np.where(self.alpha > 0)[0]

    def _kernel_rbf(self, xi, xj):
        xij = xi - xj
        if xij.ndim == 1:
            # case: x.shape = (n_features)
            return np.exp(-self.gamma * np.sum(xij**2))
        elif xij.ndim == 2:
            # case: x.shape = (n_samples, n_features)
            return np.exp(-self.gamma * np.sum(xij**2, axis=1))

    def _kernel_linear(self, xi, xj):
        xij = xi * xj
        if xij.ndim == 1:
            # case: x.shape = (n_features)
            return np.sum(xij)
        elif xij.ndim == 2:
            # case: x.shape = (n_samples, n_features)
            return np.sum(xij, 1)


class SMO:
    """Implementation of SMO.

    "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines"
    <https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf>
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/smo-book.pdf>
    """
    def __init__(self, C=10, tol=1e-3):
        self.C = C
        self.tol = tol
        self.eps = 1e-4

    def _second_lagrange_multiplier(self, one):
        if self.error[one] > 0:
            another = np.argmin(self.error)
        elif self.error[one] <= 0:
            another = np.argmax(self.error)
        return another

    def _compute_bounds(self, y1, y2, alpha1, alpha2):
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        return L, H

    def _objective_function(self, model, alpha1, alpha2,
                            K11, K12, K22, i1, i2):

        y1 = model.y[i1]
        y2 = model.y[i2]

        s = y1 * y2
        target_idx = [i for i in range(model.X.shape[0])
                      if i != i1 and i != i2]
        # v1 = 0
        # v2 = 0
        # for j in target_idx:
        #     v1 += self.y[j] * self.alpha[j] * \
        #         self.kernel_func(self.X[i1], self.X[j])
        #     v2 += self.y[j] * self.alpha[j] * \
        #         self.kernel_func(self.X[i2], self.X[j])
        v1 = model.y[target_idx] * model.alpha[target_idx] \
            * model.kernel_func(model.X[i1], model.X[target_idx])
        v1 = np.sum(v1)
        v2 = model.y[target_idx] * model.alpha[target_idx] \
            * model.kernel_func(model.X[i2], model.X[target_idx])
        v2 = np.sum(v2)

        objective = alpha1 + alpha2 - 0.5 * K11 * alpha1**2 \
            - 0.5 * K22 * alpha2**2 - s * K12 * alpha1 * alpha2 \
            - y1 * alpha1 * v1 - y2 * alpha2 * v2

        return objective

    def _take_step(self, model, i1, i2):
        if (i1 == i2):
            return 0

        # extract variables to optimization
        alpha1 = model.alpha[i1]
        x1 = model.X[i1, :]
        y1 = model.y[i1]
        E1 = self.error[i1]

        alpha2 = model.alpha[i2]
        x2 = model.X[i2, :]
        y2 = model.y[i2]
        E2 = self.error[i2]

        # calculate bounds for the new alpha2
        L, H = self._compute_bounds(y1, y2, alpha1, alpha2)
        if L == H:
            return 0

        K11 = model.kernel_func(x1, x1)
        K12 = model.kernel_func(x1, x2)
        K22 = model.kernel_func(x2, x2)
        eta = 2 * K12 - K11 - K22

        if eta < 0:
            # calculate the new alpha2
            a2 = alpha2 - y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
        else:
            Lobj = self._objective_function(model, alpha1, L,
                                            K11, K12, K22, i1, i2)
            Hobj = self._objective_function(model, alpha1, H,
                                            K11, K12, K22, i1, i2)

            # calculate the new alpha2
            if Lobj > Hobj + self.eps:
                a2 = L
            elif Lobj + self.eps < Hobj:
                a2 = H
            else:
                a2 = alpha2

        if a2 < 1e-8:
            a2 = 0
        elif a2 > self.C - 1e-8:
            a2 = self.C
        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
        # if abs(a2 - alpha2) < self.eps * a2:
            return 0

        # calculate the new alpha1
        a1 = alpha1 + y1 * y2 * (alpha2 - a2)
        # to avoid a negative value due to floating-point arithmetic
        a1 = np.clip(a1, 0, self.C)

        # update the threshold
        b_old = model.b  # hold
        model.b = b_new = self._compute_threshold(b_old, y1, y2,
                                                  E1, E2, K11, K12, K22,
                                                  a1, a2, alpha1, alpha2)

        targets = list(range(model.X.shape[0]))
        for idx, alpha_new in zip((i1, i2), (a1, a2)):
            if (alpha_new > 0) and (alpha_new < self.C):
                self.error[idx] = 0
                targets.remove(idx)

        self._update_error(model, targets, x1, x2, y1, y2,
                           a1, a2, alpha1, alpha2, b_new, b_old)

        # update alpha1 and alpha2
        model.alpha[i1] = a1
        model.alpha[i2] = a2

        # assert(0 <= a1 <= self.C)
        # assert(0 <= a2 <= self.C)

        return 1

    def _compute_threshold(self, b_old, y1, y2, E1, E2, K11, K12, K22,
                           a1, a2, alpha1, alpha2):

        b1 = - E1 - y1 * (a1 - alpha1) * K11 \
            - y2 * (a2 - alpha2) * K12 + b_old

        b2 = - E2 - y1 * (a1 - alpha1) * K12 \
            - y2 * (a2 - alpha2) * K22 + b_old

        if 0 < a1 < self.C:
            b_new = b1
        elif 0 < a2 < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2

        return b_new

    def _update_error(self, model, target_idx, x1, x2, y1, y2,
                      a1, a2, alpha1, alpha2, b_new, b_old):
        self.error[target_idx] = self.error[target_idx] \
            + y1 * (a1 - alpha1) * model.kernel_func(x1, model.X[target_idx]) \
            + y2 * (a2 - alpha2) * model.kernel_func(x2, model.X[target_idx]) \
            + b_new - b_old

    def _is_violate_kkt(self, alpha, E, y):
        r = E * y
        return (alpha > 0 and r > self.tol) or (alpha < self.C and r < -self.tol)

    def _examine_example(self, model, i2):
        y2 = model.y[i2]
        alpha2 = model.alpha[i2]
        E2 = self.error[i2]

        # check KKT condition
        if not self._is_violate_kkt(alpha2, E2, y2):
            return 0

        num_non_bounds = np.sum((model.alpha > 0) & (model.alpha < self.C))
        if (num_non_bounds > 1):
            # select the second lagrange multiplier
            i1 = self._second_lagrange_multiplier(i2)
            if self._take_step(model, i1, i2):
                return 1

        non_bounds = np.where((model.alpha > 0) & (model.alpha < self.C))[0]
        if non_bounds.shape[0] > 0:
            # select the start point randomly
            targets = np.roll(non_bounds,
                              np.random.choice(non_bounds.shape[0]))
            for i1 in targets:
                if self._take_step(model, i1, i2):
                    return 1

        targets = range(model.alpha.shape[0])
        # select the start point randomly
        targets = np.roll(targets, np.random.choice(len(targets)))
        for i1 in targets:
            if self._take_step(model, i1, i2):
                return 1

        return 0

    def solve(self, model, X, y):

        model.preset(X, y)
        self.error = -y  # f(x) - y

        num_changed = 0
        examine_all = True

        while (num_changed > 0) or examine_all:

            num_changed = 0

            if examine_all:
                for i in range(model.X.shape[0]):
                    num_changed += self._examine_example(model, i)
            else:
                mask = (model.alpha > 0) & (model.alpha < self.C)
                non_bounds = np.where(mask)[0]
                for i in non_bounds:
                    num_changed += self._examine_example(model, i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return model
