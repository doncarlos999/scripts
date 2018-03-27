#!/usr/bin/python
'''
This function will return one set of pairwise correlation values
from columns of a pandas dataframe.
Columns should be cells and rows(index) should be genes.
Correlation options = 'spearman', 'pearson'(default), 'kendal'
masking = False(default) full  matrix is used or True
half of the matrix is masked including 100% correlation line
'''

import pandas as pd
import numpy as np


def find_correlation(df=None, masking=False, correlation='pearson'):
    # make sure that dataframe is numeric
    for i in df.columns.unique():
        df[i] = pd.to_numeric(df[i])
    # produce correlation correlation matrix as np.matrix
    df_corr = df.corr(correlation).as_matrix()
    # make mask of top half of matrix to remove duplicated correlations
    mask = np.zeros_like(df_corr)
    mask[np.triu_indices_from(mask)] = True
    # make correlation correlation df
    df_corr_df = df.corr(correlation)
    # make mask df
    mask_df = pd.DataFrame(mask,
                           columns=df_corr_df.columns.values,
                           index=df_corr_df.index.values)
    # turn mask df into bool df
    bool_mask_df = mask_df == 1
    if masking is True:
        # mask out top half of matrix with nan
        df_corr_rem_dup_df = df_corr_df.mask(bool_mask_df, np.nan)
        # make nested list of all array values
        df_corr_list = df_corr_rem_dup_df.values.tolist()
    else:
        # make nested list of full matrix
        df_corr_list = df_corr_df.values.tolist()
    # squash nested list
    import itertools
    flattened_df_corr_list = list(itertools.chain(*df_corr_list))
    # drop nans from list
    df_corr_values = [x for x in flattened_df_corr_list if str(x) != 'nan']
    return df_corr_values
