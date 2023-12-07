import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

ny = 1000
sig = 40
alpha = 2

repeats = 10

true_error = []
est_error = []

denoi = lambda y: (y - 20) * (y > 20)

alpha_range = np.linspace(0.3, 10, 1000)

for alpha in alpha_range:
    alpha_sq = alpha ** 2
    a = ((1 + alpha_sq) / alpha_sq)
    yest_err = 0
    yest_err_approx = 0
    for r in range(repeats):
        y0 = np.random.uniform(20, 100, ny) * np.random.binomial(1, 0.8, ny)
        N = np.random.normal(0, sig, ny)
        Ntilde = np.random.normal(0, alpha * sig, ny)

        y = y0 + N
        ytild = y + Ntilde

        yest = ((1 + alpha_sq) * denoi(ytild) - ytild) / alpha_sq

        yest_err += np.sum((yest - y0)**2)
        yest_err_approx += np.sum((a**2 * (denoi(ytild) - ytild)) ** 2) - ny * sig**2 * a

        print(np.sum((a * (denoi(ytild) - ytild)) ** 2), ny * sig**2 * a)

        #print(np.sum((a * (denoi(ytild) - ytild)) ** 2) - ny * sig**2 * a, a * 2 * np.inner(denoi(ytild), ( Ntilde/ alpha_sq - N )))
        yest_err_approx += a * 2 * np.inner(denoi(ytild), ( Ntilde/ alpha_sq - N )  )

    true_error.append(yest_err / repeats)

    est_error.append(yest_err_approx / repeats)

true_error = np.array(true_error)
est_error = np.array(est_error)

print('True minimum is at {}'.format(alpha_range[np.argmin(true_error)]))
print('Estimates minimum is at {}'.format(alpha_range[np.argmin(est_error)]))

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.plot(alpha_range, true_error)
plt.title('True error')
plt.ylim([0, 1.7e8])
plt.subplot(132)
plt.plot(alpha_range, est_error)
plt.title('Est error')
plt.ylim([0, 1.7e8])
plt.subplot(133)
plt.plot(alpha_range, 100 * (np.array(true_error) - np.array(est_error)) / np.array(true_error) )
plt.title('% difference')
plt.show()
