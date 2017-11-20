import utils
import numpy as np

gts = [[-19.8, -4.4, 0], [-40.3, 7.8, 0], [-65.4, 4.5, 0], [17.4, -10.0, 0], [-54.6, -11.1, 0], [62.1, -4.6, 0], [-127.9, 4.6, 0], [-17.8, -.6, 0], [-15.2, 0.4, 0], [-37.2, -14.9, 0], [-23.6, -15.4, 0], [19.8, -5.7, 0], [25.8, 3.0, 0], [-67.6, 12.4, 0]]
preds = [[-19.6, -9.1, 0], [-37.3, 2.8, 0], [-57.7, -3.4, 0], [21.3, 4.1, 0], [-55.1, -1.6, 0], [78.9, 3.2, 0], [-113.9, 4.9, 0], [-24.5, -.2, 0], [-11.9, -2.4, 0], [-43.9, -18.0, 0], [-20.7, -3.3, 0], [17.2, -2.7, 0], [22.6, 7.2, 0], [-61.5, -1.4, 0]]
true_errs = [4.6, 5.8, 10.9, 14.0, 12.7, 16.8, 14.0, 6.6, 4.3, 7.4, 12.4, 3.9, 5.3, 15.1]

print(len(gts))
print(len(preds))
print(len(true_errs))
for gt, pred, true_err in zip(gts, preds, true_errs):
    err1 = np.linalg.norm(np.array(utils.relative_rotation(gt, pred)))
    err2 = utils.average_angular_error(np.array([gt]), np.array([pred]))
    print('Method 1 error: %5.1f     Method 2 error: %5.1f      GT error: %5.1f'%(err1, err2, true_err))
