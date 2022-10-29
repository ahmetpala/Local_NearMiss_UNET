import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats as st

from batch.data_transforms.ahmet_db_with_limits import xr_db_with_limits
from batch.label_transforms.ahmet_refine_label_boundary import P_refine_label_boundary


# This class should be added to the DataReaderZarr class as a function
class GeneratePatchStat:
    def __init__(self, survey, coordinates, quantiles=[0, 0.25, 0.5, 0.75, 0.95, 1], patch_size=(256, 256)):
        # coordinates = 2d array containing center x,y values
        # x, y, mean, median, n_pixels_below_seabed, Nr of fish categories + ignore + background
        self.patch_size = patch_size
        self.coordinates = coordinates
        self.data = np.zeros((len(self.coordinates), 53))  ## UPDATE EDILECEK
        self.quantiles = quantiles
        self.survey = survey
        self.stats = None

    def calculate_stats(self, only_pixel_counts=True):
        for i, (x, y) in tqdm(enumerate(self.coordinates), total=len(self.coordinates)):
            # Data Loading & Preparations
            org_sv = self.survey.get_data_slice(idx_ping=x - self.patch_size[0] // 2, n_pings=self.patch_size[0],
                                                idx_range=int(y - self.patch_size[1] // 2), n_range=self.patch_size[1],
                                                return_numpy=False, frequencies=[38000, 200000])
            data_Sv = xr_db_with_limits(org_sv)
            seabed_mask = self.survey.get_seabed_mask(idx_ping=x - self.patch_size[0] // 2, n_pings=self.patch_size[0],
                                                      idx_range=int(y - self.patch_size[1] // 2), n_range=self.patch_size[1],
                                                      return_numpy=False)
            y_label = self.survey.get_label_slice(idx_ping=x - self.patch_size[0] // 2, n_pings=self.patch_size[0],
                                                  idx_range=int(y - self.patch_size[1] // 2), n_range=self.patch_size[1],
                                                  return_numpy=True)
            modified = P_refine_label_boundary(ignore_zero_inside_bbox=False,
                                               threshold_val=[1e-07, 1e-04], frequencies=[38, 200])(data=org_sv.values,
                                                                                                    labels=y_label,
                                                                                                    echogram=org_sv.values)
            modified = modified[1]  # Final 2D modified annotations
            np_Sv = data_Sv.sel(frequency=38000).values  # Sv data slice in 38 kHz frequency

            # Coordinates
            self.data[i, 0] = x
            self.data[i, 1] = y

            if only_pixel_counts is not True:
                # Patch Statistics
                self.data[i, 2] = mean_Sv = np.nanmean(np_Sv)  # mean (excluding nan)
                self.data[i, 3] = mode_Sv = st.mode(np_Sv.flatten())[0]  # mode (excluding nan)
                self.data[i, 4] = np.nanstd(np_Sv)  # standard deviation (excluding nan)
                self.data[i, 5], self.data[i, 6], self.data[i, 7], self.data[i, 8], self.data[i, 9], self.data[i, 10] = np.nanquantile(np_Sv, q=self.quantiles)

            # Number of Pixels
            self.data[i, 11] = seabed_mask.sum().values  # Pixels below seabed
            self.data[i, 12] = len(modified[modified == 1])  # Other class
            self.data[i, 13] = len(modified[modified == 27])  # Sandeel class
            self.data[i, 14] = len(modified[modified == 6009])  # Possible Sandeel class
            self.data[i, 15] = np.isnan(np_Sv).sum()  # Number of nan pixels
            self.data[i, 16] = 256 * 256 - (
                        self.data[i, 12] + self.data[i, 13] + self.data[i, 14] + self.data[i, 15])  # Background class (Excluding nan values)

        # Class Statistics
        if only_pixel_counts is not True:
            # Other Class
            if self.data[i, 12] != 0:
                ot_filt = np_Sv[modified == 1]
                # Mean, Mode, Std Other Sv
                self.data[i, 17], self.data[i, 18], self.data[i, 19] = ot_filt.mean(), st.mode(ot_filt.flatten())[0], np.std(ot_filt)
                # 0 0.25 0.5 0.75 0.95 1 Quantiles
                self.data[i, 20], self.data[i, 21], self.data[i, 22], self.data[i, 23], self.data[i, 24], self.data[i, 25] = np.quantile(ot_filt, q=self.quantiles)

            # Sandeel Class
            if self.data[i, 13] != 0:
                sd_filt = np_Sv[modified == 27]
                # Mean, Mode, Std Sandeel Sv
                self.data[i, 26], self.data[i, 27], self.data[i, 28] = sd_filt.mean(), st.mode(sd_filt.flatten())[0], np.std(sd_filt)
                # 0 0.25 0.5 0.75 0.95 1 Quantiles
                self.data[i, 29], self.data[i, 30], self.data[i, 31], self.data[i, 32], self.data[i, 33], self.data[i, 34] = np.quantile(sd_filt, q=self.quantiles)

            # Possible Sandeel Class
            if self.data[i, 14] != 0:
                ps_filt = np_Sv[modified == 6009]
                # Mean, Mode, Std possible Sandeel Sv
                self.data[i, 35], self.data[i, 36], self.data[i, 37] = ps_filt.mean(), st.mode(ps_filt.flatten())[0], np.std(ps_filt)
                # 0 0.25 0.5 0.75 0.95 1 Quantiles
                self.data[i, 38], self.data[i, 39], self.data[i, 40], self.data[i, 41], self.data[i, 42], self.data[i, 43] = np.quantile(sd_filt, q=self.quantiles)

            # Background
            if self.data[i, 16] == 256 * 256:
                # If there is no fish, statistics are the same as patch
                self.data[i, 44], self.data[i, 45], self.data[i, 46] = self.data[i, 2], self.data[i, 3], self.data[i, 4]
                self.data[i, 47], self.data[i, 48], self.data[i, 49], self.data[i, 50], self.data[i, 51], self.data[i, 52] = self.data[i, 20], self.data[i, 21], self.data[
                    i, 22], self.data[i, 23], self.data[i, 24], self.data[i, 25]
            else:
                bg_flt = np_Sv[np.logical_or((modified == 0), (modified == -1))]
                # Mean, Mode, Std Background Sv
                self.data[i, 44], self.data[i, 45], self.data[i, 46] = np.nanmean(bg_flt), st.mode(bg_flt.flatten(), nan_policy='omit')[
                    0], np.nanstd(bg_flt)
                # 0 0.25 0.5 0.75 0.95 1 Quantiles
                self.data[i, 47], self.data[i, 48], self.data[i, 49], self.data[i, 50], self.data[i, 51], self.data[i, 52] = np.nanquantile(bg_flt,
                                                                                                              q=self.quantiles)

        self.data[:, 16] = 256 * 256 - (self.data[:, 12] + self.data[:, 13] + self.data[:, 14] + self.data[:, 15])  # nop_bg correction

        # Converting array to pandas df
        pdata = pd.DataFrame(self.data)

        names_modified = ['x', 'y', 'mean_Sv', 'mode_Sv', 'std_Sv',
                          '0_Sv', '25_Sv', '50_Sv', '75_Sv', '95_Sv', '100_Sv',
                          'nop_below_seabed', 'nop_other', 'nop_sandeel', 'nop_possandeel', 'nop_nan', 'nop_background',
                          'mean_other', 'mode_other', 'std_other',
                          '0_Other', '25_Other', '50_Other', '75_Other', '95_Other', '100_Other',
                          'mean_sandeel', 'mode_sandeel', 'std_sandeel',
                          '0_sandeel', '25_sandeel', '50_sandeel', '75_sandeel', '95_sandeel', '100_sandeel',
                          'mean_psandeel', 'mode_psandeel', 'std_psandeel',
                          '0_psandeel', '25_psandeel', '50_psandeel', '75_psandeel', '95_psandeel', '100_psandeel',
                          'mean_bg', 'mode_bg', 'std_bg',
                          '0_bg', '25_bg', '50_bg', '75_bg', '95_bg', '100_bg']
        pdata.columns = names_modified

        # Class Label Assignment
        pdata['class'] = 'background'
        pdata.loc[(pdata['nop_other'] > 0) & (pdata['nop_sandeel'] == 0)
                  & (pdata['nop_below_seabed'] == 0), 'class'] = 'other'
        pdata.loc[(pdata['nop_other'] == 0) & (pdata['nop_sandeel'] > 0)
                  & (pdata['nop_below_seabed'] == 0), 'class'] = 'sandeel'
        pdata.loc[(pdata['nop_other'] == 0) & (pdata['nop_sandeel'] == 0)
                  & (pdata['nop_below_seabed'] > 0) & (pdata['nop_below_seabed'] != 256 * 256), 'class'] = 'seabed'
        pdata.loc[(pdata['nop_other'] == 0) & (pdata['nop_sandeel'] > 0)
                  & (pdata['nop_below_seabed'] > 0) & (
                              pdata['nop_below_seabed'] != 256 * 256), 'class'] = 'seabed_sandeel'
        pdata.loc[(pdata['nop_other'] > 0) & (pdata['nop_sandeel'] == 0)
                  & (pdata['nop_below_seabed'] > 0) & (
                              pdata['nop_below_seabed'] != 256 * 256), 'class'] = 'seabed_other'
        pdata.loc[(pdata['nop_other'] > 0) & (pdata['nop_sandeel'] > 0)
                  & (pdata['nop_below_seabed'] == 0), 'class'] = 'sandeel_other'
        pdata.loc[(pdata['nop_other'] > 0) & (pdata['nop_sandeel'] > 0)
                  & (pdata['nop_below_seabed'] > 0) & (
                              pdata['nop_below_seabed'] != 256 * 256), 'class'] = 'seabed_sandeel_other'
        self.stats = pdata
        return pdata


