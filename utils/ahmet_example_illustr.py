# Example Illustration for a patch - Echogram, annotations, Background Sv Distribution
import numpy as np
from matplotlib import pyplot as plt

from batch.data_transforms.ahmet_db_with_limits import xr_db_with_limits
from data.normalization import db


def examp_illustr(survey, x, y, fig_start=-75, fig_finish=5, s_bin=5, db_w_limit=True, patch_size=[256, 256]):
    c = survey.get_label_slice(idx_ping=x - patch_size[0] // 2, n_pings=patch_size[0],
                               idx_range=int(y - patch_size[1] // 2), n_range=patch_size[1],
                               return_numpy=True)
    if db_w_limit:
        a = xr_db_with_limits(survey.get_data_slice(idx_ping=x - patch_size[0] // 2, n_pings=patch_size[0],
                                                    idx_range=int(y - patch_size[1] // 2), n_range=patch_size[1],
                                                    return_numpy=False, frequencies=[38000]))
        echo = a[0, :, :].values
        bg = a[0].values[np.logical_or(c == -1, c == 0)]  # Background

    else:
        a = survey.get_data_slice(idx_ping=x - patch_size[0] // 2, n_pings=patch_size[0],
                                  idx_range=int(y - patch_size[1] // 2), n_range=patch_size[1],
                                  return_numpy=False, frequencies=[38000])
        echo = db(a[0, :, :].values)
        bg = db(a[0].values[np.logical_or(c == -1, c == 0)])  # Background

    print('mean background Sv =', bg.mean(), ' median background Sv =', np.median(bg))
    print('mean Sandeel Sv =', a[0].values[c == 27].mean(), ' median Sandeel Sv =', np.median(a[0].values[c == 27]))
    # Echogram Plot
    plt.figure(figsize=(12, 12))
    plt.suptitle('Patch Visualization at x=' + str(x) + ' and y=' + str(y))

    plt.subplot(2, 2, 1)
    plt.title('Echogram')
    plt.imshow(echo.T)
    plt.colorbar()

    # Annotation Plot
    plt.subplot(2, 2, 3)
    plt.title('Annotations')
    plt.imshow(c.T)
    plt.colorbar()

    # Background Histogram Plot
    plt.subplot(2, 2, 2)
    plt.title('Background Sv Histogram')
    plt.hist(bg.flatten(), bins=np.arange(fig_start, fig_finish, s_bin), log=False)
    plt.xlabel('sv')
    plt.ylabel('Counts')
    plt.axvline(x=bg.mean(), color='black', label='mean')
    plt.axvline(x=np.median(bg), color='green', label='median')
    plt.axvline(x=np.nanpercentile(bg, 25), color='lightgray', label='25 Percentile')
    plt.axvline(x=np.nanpercentile(bg, 75), color='dimgray', label='75 Percentile')
    plt.legend()

    # Sandeel Histogram Plot
    plt.subplot(2, 2, 4)
    plt.title('Sandeel Sv Histogram')
    plt.hist(echo[c == 27].flatten(),
             bins=np.arange(fig_start, fig_finish, s_bin), log=False)
    plt.xlabel('Sv')
    plt.ylabel('Counts')
    plt.axvline(x=(echo[c == 27].flatten()).mean(), color='red', label='Sandeel mean Sv')
    plt.axvline(x=np.median((echo[c == 27].flatten())), color='orange', label='Sandeel median Sv')
    plt.axvline(x=np.nanpercentile(echo[c == 27].flatten(), 25), color='lightgray', label='25 Percentile')
    plt.axvline(x=np.nanpercentile(echo[c == 27].flatten(), 75), color='dimgray', label='75 Percentile')
    plt.legend()
    plt.show()