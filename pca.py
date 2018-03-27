def make_2d_pca(data, marker=False, scaling=False):
    '''make 2d PCA plot from pandas dataframe with genes as columns
    rows as samples. Cell type as a column "cell". Returns seaborn lmplot
    Scaling options = AutoScale, MinMax, MaxAbs, Robust
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    matrix_df = data.drop('cell', axis='columns')
    matrix = matrix_df.as_matrix()
    if scaling == 'AutoScale':
        matrix = preprocessing.scale(matrix)
    if scaling == 'MinMax':
        min_max_scaler = preprocessing.MinMaxScaler()
        matrix = min_max_scaler.fit_transform(matrix)
    if scaling == 'MaxAbs':
        max_abs_scaler = preprocessing.MaxAbsScaler()
        matrix = max_abs_scaler.fit_transform(matrix)
    if scaling == 'Robust':
        robust_scaler = preprocessing.RobustScaler()
        matrix = robust_scaler.fit_transform(matrix)
    if scaling is False:
        matrix = matrix
    pca = PCA(n_components=2)
    pca.fit(matrix)
    print('explained variance is {0}'.format(
        pca.explained_variance_ratio_))
    pca = PCA(n_components=2).fit_transform(matrix)

    pca_df = pd.DataFrame(matrix,
                          columns=matrix_df.columns.values,
                          index=matrix_df.index.values)

    pca_df['PCA1'] = pca[:, 0]
    pca_df['PCA2'] = pca[:, 1]
    pca_df = pca_df.join(data['cell'])

    markers = markers = ['1', '2', '3', '4', 'p', 's',
                         'x', 'o', '.', 's', '+']
    sns.set_palette(sns.color_palette("hls",
                                      pca_df['cell'].unique().size))
    y = []
    if marker is True:
        x = sns.lmplot("PCA1", "PCA2",
                       data=pca_df,
                       hue='cell',
                       fit_reg=False,
                       markers=markers[0:pca_df['cell'].unique().size])
        y.append(x)
    else:
        x = sns.lmplot("PCA1", "PCA2",
                       data=pca_df,
                       hue='cell',
                       fit_reg=False)
        y.append(x)
    y.append(pca_df)
    return y
