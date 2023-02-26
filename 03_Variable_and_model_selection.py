# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:33:50 2022

@author: SEZE
"""

###############################################################################
#                                                                             #
#                            3. Model selection                               #
#                                                                             #
###############################################################################
## Import libraries
try:
    lib = __import__('A_Libraries')
except:
    raise Exception('\n0.A - Libraries loaded - FAILED')
os, pd, StandardScaler, sc, PCA, np, plt, KMeans, mplot3d, sn, rc, matlib, itertools, linalg, mpl, mixture, DBSCAN, metrics, make_blobs, warnings = lib.Libraries()




def variance_explained(df, threshold = 0.55):
    print(f"\n3.1 - Variance explained {threshold}")
    # df for test of PCA
    
    sc.fit(df)
    X_train_std = sc.transform(df)
    #X_test_std = sc.transform(X_test)
    #
    # Instantiate PCA
    pca = PCA(n_components = threshold)
    #pca = PCA()
    
    # Determine transformed features
    #
    X_train_pca = pca.fit_transform(X_train_std)
    
    exp_var_pca = pca.explained_variance_ratio_
    #
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    #
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    #
    # Create the visualization plot
    #
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return X_train_pca, pca, X_train_std


def k_means(X_train_pca, clusters = 4):
    print(f"\n3.2 - K-means, with: {clusters} clusters")
    # Perform a simple k-means
    
    data = X_train_pca

    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/elbow_curve_k_means', dpi = 600)
    plt.show()

    # Number of clusters
    n = clusters

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data)
    
    # Check dimensions of data
    if data.shape[1] == 1:
        x = data[:,0]
        y = np.zeros(data.shape[0])
    else:
        x = data[:,0]
        y = data[:,1]
        #z = data[:,2]
    l = range(n)




    plt.title('K-means')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')


    plt.scatter(x, y, c=kmeans.labels_, cmap=mpl.colormaps['Paired'])
    scatter = plt.scatter(x, y, c=kmeans.labels_, cmap=mpl.colormaps['Paired'])
    plt.legend(handles=scatter.legend_elements()[0], 
               labels=l,
               title="Classes",
               loc='upper right')

    plt.savefig(f'C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/k_means_{n}_clusters', dpi = 600)
    #plt.savefig('C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/k_means_5_clusters_3d', dpi = 900)
    #plt.rcParams['figure.dpi'] = 600
    plt.show()

    return kmeans


# Intermediate plotting tool for PC variable effects
def myplot(score,coeff,label,var,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c=label, cmap=mpl.colormaps['Paired'])
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            #plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            plt.text(coeff[i,0]* 1.1, coeff[i,1] * 1.1, var[i], color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.1, coeff[i,1] * 1.1, labels[i], color = 'g', ha = 'center', va = 'center')


# Variables effect on the principal components
def pc_var_effects(X_train_pca, pca,label,var,x=1,y=2):
    print("\n3.3 - Variable effects on PC")
    
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(x))
    plt.ylabel("PC{}".format(y))
    plt.grid()
    
    #Call the function. Use only the 2 PCs.
    myplot(X_train_pca[:,0:2],np.transpose(pca.components_[0:2, :]),label,var)
    plt.savefig('C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/Variable_effects_on_PC1_PC2', dpi = 600)
    plt.show()
    return

def df_id_add(df, df_peaks):
    peak_l = []
    spec_id_l = []


    for i in df.index:
            peak_l.append(df_peaks['label'][i])
            spec_id_l.append(df_peaks['spec_id'][i])
        

    df['peak_n'] = peak_l
    df['spec_id'] = spec_id_l

    # Variables in data
    var = df.columns
    return var, df



def correlation_matrix(df, var, X_train_std):
    print("\n3.4 - Correlation matrix")
    # Correlation matrix
    c=3
    df_corr = df.iloc[:,:-c]

    std_var = var[:-c]
    df_std = pd.DataFrame(X_train_std, columns = std_var)
    corrMatrix_std = df_std.corr()
    f, ax = plt.subplots(figsize=(30, 15))
    corr_map = sn.heatmap(corrMatrix_std, annot=True, cmap='YlGn',linewidths=.5)

    plt.savefig('C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/Correlation_matrix', dpi = 600)

    plt.show()
    return


def label_predictions(df,label, title):
    print("\n3.5 - Label predictions, plotted on rention and ID")
    
    # Scatter plot based on vial and label
    #n = df['prediction'].max()+1
    n = len(set(label))
    y = df['spec_id']
    x = df['retention']
    #z = data[:,2]
    l = range(n)
    
    scatter = plt.scatter(x, y, c=label, cmap=mpl.colormaps['Paired'])
    
    plt.title(f'{title}')
    plt.ylabel('Vial ID')
    plt.xlabel('Retention time - Normalized')
    plt.scatter(x, y, c=label, cmap=mpl.colormaps['Paired'])
    #plt.legend(handles=scatter.legend_elements()[0], 
    #           labels=l,
    #           title="Classes",
    #           loc='center right')
    plt.savefig(f'C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/Predictions_{n}_clusters_ID_Retention', dpi = 600)
    plt.show()
    return
    
def format_axes(ax, **kwargs):
    
    """
    Format a font to my standards
    input: axes and key word arguments
    param: ax - axes to be formatted
    param: border - boolean, True = grey border, False = None. Default True
    """
    
    rc('font', family = 'serif')
    
    # Set border
    border = True
    if 'border' in kwargs:
        border = kwargs['border']
        
    if border:
        ax.spines['top'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.spines['left'].set_color('grey')
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    # Format label and tick fonts
    tickwidth = 1
    if border == False:
        tickwidth = 2
    
    ax.xaxis.label.set_size(16)
    ax.xaxis.label.set_color('grey')
    ax.yaxis.label.set_size(16)
    ax.yaxis.label.set_color('grey')
    ax.title.set_color('grey')
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor = 'grey')
    
    return ax

def soft_clustering_weights(data, cluster_centres, **kwargs):
    
    """
    Function to calculate the weights from soft k-means
    data: Array of data. Features arranged across the columns with each row being a different data point
    cluster_centres: array of cluster centres. Input kmeans.cluster_centres_ directly.
    param: m - keyword argument, fuzziness of the clustering. Default 2
    """
    
    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    if 'm' in kwargs:
        m = kwargs['m']
    
    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]
    Nfeatures = data.shape[1]

    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:,i] = np.sum((data-np.matlib.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)
    

    
    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist**(2/(m-1))*np.matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
    Weight = 1./invWeight
    
    return Weight



def plotter_to_use(x1, y1, x2, y2, n_classes, l1, l2, upper_bound, lower_bound,  ex_labels, ex_peaks_l_1, ex_peaks_count_1, ex_peaks_l_2, ex_peaks_count_2, t1 = False, t2 = False, x_label = False, y_label = False, cc = False, **kwargs):
    
    fig, axs = plt.subplots(1, 2, figsize = (15,7))

    ax = format_axes(axs[0])
    
    colors = ['blue', 'brown']
    labels = ['Dimer', 'Main']
    for i in set(l1):
        ax.scatter(
            x1[l1 == i],
            y1[l1 == i],
            marker = '1',
            label = labels[i],
            cmap = mpl.colormaps['Paired']
            )
    """
    ax.scatter(
        x1,
        y1,
        #c=kmeans.labels_,
        c=l1,
        cmap=mpl.colormaps['Paired'], 
        marker = '1',
        label = 'Main'
    )
    """
    markers = ['2', 'v', 'o']
    for i in range(len(ex_labels)):
        ax.scatter(ex_peaks_l_1[i]['spec_id'], ex_peaks_l_1[i]['retention'], color = 'grey', marker = markers[i], label = f'{ex_labels[i]} - {ex_peaks_count_1[i]}')
        
    

    if upper_bound != 0:
        ax.axhline(y = upper_bound, linestyle='dashed', color='gray')
    if lower_bound != 0:
        ax.axhline(y = lower_bound, linestyle='dashed', color='gray')
    
    # Labeling
    if x_label and y_label == False:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)       
    
    # Title
    if t1 == False:
        ax.set_title('', pad = 20, fontsize = 18)
    else:
        ax.set_title(t1, pad = 20, fontsize = 18)
    ax.set_ylim([1,7])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=False, title = 'Classes')
    
    ax = format_axes(axs[1])

    for i in set(l1):
        ax.scatter(
            x1[l1 == i],
            y1[l1 == i],
            marker = '1',
            label = labels[i],
            cmap = mpl.colormaps['Paired']
            )
    
    markers = ['2', 'v', 'o']
    for i in range(len(ex_labels)):
        ax.scatter(ex_peaks_l_2[i]['spec_id'], ex_peaks_l_2[i]['retention'], color = 'grey', marker = markers[i], label = f'{ex_labels[i]} - {ex_peaks_count_2[i]}')
    

    if upper_bound != 0:
        ax.axhline(y = upper_bound, linestyle='dashed', color='gray')
    if lower_bound != 0:
        ax.axhline(y = lower_bound, linestyle='dashed', color='gray')
    #for i in range(len(n_classes)):
    #    ax.plot(x2,y2, '.r-')



    if cc != False:
        
        ax.plot(
            cc[:,-2],
            cc[:,-1],
            'xk',
            markersize = 15,
            markeredgewidth = 3
        )
        
    if x_label and y_label == False:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)       
        
    if t2 == False:
        ax.set_title('', pad = 20, fontsize = 18)
    else:
        ax.set_title(t2, pad = 20, fontsize = 18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=False, title = 'Classes')
    #fig.savefig('Figures/confident_kmeans.png', dpi = 500, bbox_inches = 'tight')

    fig.savefig(f'C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Results/{t1}_w2.jpg', dpi = 600)
    
    ax.set_ylim([1,7])
    
    plt.show()    
    
    return


def plotter_to_use_old(x1, y1, x2, y2, n_classes, l1, l2, upper_bound, lower_bound, df_peaks_ex, t1 = False, t2 = False, x_label = False, y_label = False, cc = False, **kwargs):
    
    fig, axs = plt.subplots(1, 2, figsize = (15,7))

    ax = format_axes(axs[0])

    ax.scatter(
        x1,
        y1,
        #c=kmeans.labels_,
        c=l1,
        cmap=mpl.colormaps['Paired'], marker = '1'
    )
    
    ax.scatter(df_peaks_ex['spec_id'],
               df_peaks_ex['retention'],
               color = 'grey',
               marker = '2')
    
    if upper_bound != 0:
        ax.axhline(y = upper_bound, linestyle='dashed', color='gray')
    if lower_bound != 0:
        ax.axhline(y = lower_bound, linestyle='dashed', color='gray')
    
    # Labeling
    if x_label and y_label == False:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)       
    
    # Title
    if t1 == False:
        ax.set_title('', pad = 20, fontsize = 18)
    else:
        ax.set_title(t1, pad = 20, fontsize = 18)
    ax.set_ylim([1,7])

    ax = format_axes(axs[1])

    ax.scatter(
        x2,
        y2,
        c = l2,
        cmap=mpl.colormaps['Paired'], marker = '1'
    )
    
    ax.scatter(df_peaks_ex['spec_id'],
               df_peaks_ex['retention'],
               color = 'grey',
               marker = '2')
    
    if upper_bound != 0:
        ax.axhline(y = upper_bound, linestyle='dashed', color='gray')
    if lower_bound != 0:
        ax.axhline(y = lower_bound, linestyle='dashed', color='gray')
    #for i in range(len(n_classes)):
    #    ax.plot(x2,y2, '.r-')



    if cc != False:
        
        ax.plot(
            cc[:,-2],
            cc[:,-1],
            'xk',
            markersize = 15,
            markeredgewidth = 3
        )
        
    if x_label and y_label == False:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)       
        
    if t2 == False:
        ax.set_title('', pad = 20, fontsize = 18)
    else:
        ax.set_title(t2, pad = 20, fontsize = 18)

    #fig.savefig('Figures/confident_kmeans.png', dpi = 500, bbox_inches = 'tight')

    fig.savefig(f'C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/{t1}_w2.jpg', dpi = 600)
    
    ax.set_ylim([1,7])
    
    plt.show()    
    
    return
    

def confident_k_means(df, classes, X_train_pca,kmeans, tolerance = False, confidence = 0.95):
    print("\n3.6 - Confident label selection")
    # New test
    # Check dimensions of data
    if X_train_pca.shape[1] == 1:
        X_train_pca = np.array([X_train_pca[:,0], np.zeros(X_train_pca.shape[0])]).T
        kmeans.cluster_centers_ = np.array([kmeans.cluster_centers_.T[0], np.zeros(kmeans.cluster_centers_.shape[0])]).T
    
    labels = ['Dimer', 'Main']
    colors = ['orange', 'deepskyblue']
    c_centers_l = []
    for i in range(classes):
        df['p' + str(i)] = 0
        c_centers_l.append('p'+str(i))
    df[c_centers_l] = soft_clustering_weights(X_train_pca, kmeans.cluster_centers_)

    df['confidence'] = np.max(df[c_centers_l].values, axis = 1)

    df[['PC 1', 'PC 2']] = X_train_pca     
            

    fig, axs = plt.subplots(1, 2, figsize = (15,7))

    ax = format_axes(axs[0])
    
    for i in range(classes):
        ax.scatter(
            #X_train_pca[:,0],
            #X_train_pca[:,1],
            df['PC 1'][df['prediction'] == i],
            df['PC 2'][df['prediction'] == i],
            #c=kmeans.labels_,
            #c=df['prediction'][df['prediction'] == i],
            cmap=mpl.colormaps['Paired_r'],
            #color = colors[i],
            label = labels[i],
            marker = '1'
        )
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=False, title = 'Classes')
    ax.set_xlabel('Rt - Normalized')
    ax.set_ylabel('Peak intensity - Normalized')
    ax.set_title('k-means classification', pad = 20, fontsize = 18)



    ax = format_axes(axs[1])
    for i in range(classes):
        ax.scatter(
            df.loc[df['confidence'] > confidence, 'PC 1'][df['prediction'] == i],
            df.loc[df['confidence'] > confidence, 'PC 2'][df['prediction'] == i],
            #c = df.loc[df['confidence'] > confidence, 'prediction'][df['prediction'] == i],
            cmap=mpl.colormaps['Paired'],
            #color = colors[i],
            label = labels[i],
            marker = '1'
        )
        
    """    
    ax.scatter(
        df.loc[df['confidence'] > confidence, 'PC 1'],
        df.loc[df['confidence'] > confidence, 'PC 2'],
        c = df.loc[df['confidence'] > confidence, 'prediction'],
        cmap=mpl.colormaps['Paired']
    )
    """



    ax.plot(
        kmeans.cluster_centers_[:,-2],
        kmeans.cluster_centers_[:,-1],
        'xk',
        markersize = 15,
        markeredgewidth = 3
    )

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=False, title = 'Classes')
    ax.set_xlabel('Rt - Normalized')
    ax.set_ylabel('Peak intensity - Normalized')
    ax.set_title('Confident k-means classification', pad = 20, fontsize = 18)

    #fig.savefig('Figures/confident_kmeans.png', dpi = 500, bbox_inches = 'tight')

    fig.savefig(f'C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/Confident_K_Means_with_{confidence}_confidence.jpg', dpi = 600)

    plt.show()

    df_confident = df.loc[df['confidence'] > confidence]
    
    
    # Check if their is multiple peaks in the same chromatograms assigned to the same class
    ## if so, select the one with the highest confidence
    if tolerance == False:
        #for i in range(len(df_confident)):
            
        # Select the result with the highest confidence if multiple peaks in same chromatogram has been labeled the same
        
        #res = df_confident.reset_index()
        #res = pd.DataFrame({'confidence' : df_confident.groupby(['spec_id','prediction'])['confidence'].max()}).reset_index()
        #res = pd.DataFrame({'confidence' : df_confident.groupby(['spec_id','prediction'])['confidence'].transform('max')})
        #df_confident['index'] = df_confident.index
        idx = df_confident.groupby(['spec_id','prediction'])['confidence'].transform(max) == df_confident['confidence']
        res = df_confident[idx]
        #(df_confident['spec_id'] in res['spec_id']) &
        #t_df = df_confident[(df_confident[['prediction','spec_id']].isin(res[['spec_id','prediction']]))]
        
        
        #res = df_confident.groupby(['spec_id','prediction'])
    else:
        res = df_confident
    #res = 0
    return df_confident, res




def plotter(x,y,c):
    print("\n3.7 - Label predictions, plotted on Rention time and Area fraction")
    
    # Scatter plot based on vial and label
    #z = data[:,2]
    n = c.max()+1
    l = range(n)
    
    scatter = plt.scatter(x, y, c=c, cmap=mpl.colormaps['Paired'])
    
    plt.title(f'K-means with {n} classes')
    plt.xlabel(f'{x.name}')
    plt.ylabel(f'{y.name}')
    plt.scatter(x, y, c=c, cmap=mpl.colormaps['Paired'])
    #plt.legend(handles=scatter.legend_elements()[0], 
    #           labels=l,
    #           title="Classes",
    #           loc='center right')
    plt.savefig(f'C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/Predictions_{n}_{x.name}_{y.name}', dpi = 600)
    plt.show()
    return
    

def GMMBIC(df):
    X = df
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 4)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    colors = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange",'purple','lightgreen'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    plt.title("BIC score per model")
    xpos = (
        np.mod(bic.argmin(), len(n_components_range))
        + 0.65
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, colors)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(
        f"Selected GMM: {best_gmm.covariance_type} model, "
        f"{best_gmm.n_components} components"
    )
    plt.subplots_adjust(hspace=0.35, bottom=0.02)
    plt.show()
    return




def DBSCANNER(X, labels_true, stats = True):
    print("\n3.9 - DBSCAN - Label predictions, plotted on Rention time and Area fraction")
    #db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    db = DBSCAN(min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    if stats == True:
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print(
            "Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels)
        )
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    
    
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Paired(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = labels == k
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=8,
        )
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=8,
        )
    plt.xlabel('Retention time - normalized')
    plt.ylabel('Area fraction - normalized')
    plt.title("DBSCAN with %d classes" % n_clusters_)
    plt.show()
    
    
    return db



def Dirichlet_GMM(X):
    print("\n3.10 - DPPGMM - Label predictions, plotted on Rention time and Area fraction")
    # Fit a Dirichlet process Gaussian mixture using ten components
    dpgmm = mixture.BayesianGaussianMixture(n_components=10, covariance_type="full").fit(X)
    Y_ = dpgmm.predict(X)
    proba = dpgmm.predict_proba(X)
    means = dpgmm.means_
    covariances = dpgmm.covariances_
    index = 1
    
    #splot = plt.subplot(2, 1, 1 + index)
    splot = plt.subplot()
    classes = 0
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.

        colors=mpl.colormaps['Paired']
        if not np.any(Y_ == i):
            continue
        
        X_in_class = X[Y_ == i]
        #plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)
        plt.scatter(X_in_class[:, 0], X_in_class[:, 1], 0.8, color=colors(i))

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=colors(i))
        q = ell.contains_points(X_in_class)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.4)

        splot.add_artist(ell)
        classes += 1

    title = f"Bayesian GMM with Dirichlet PP with {classes} classes"

    plt.xlabel('Retention time - Normalized')
    plt.ylabel('Area fraction')
    plt.title(title)
    plt.savefig('C:/Users/SEZE/Desktop/Master_thesis_2022/Figures/Temp/Bayesian_GMM_with_Dirichlet.jpg', dpi = 600)
    plt.show()
    return ell, dpgmm, Y_, proba


def DTW_(df, file_path):
    
    
    # -*- coding: utf-8 -*-
    """
    Created on Mon Feb 13 17:00:54 2023

    @author: SEZE
    """




    import json
    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import matplotlib.pyplot as plt
    import numpy as np
    #plt.rcParams['figure.dpi'] = 600
    # Import file
    file_path = open(file_path)
    #file_path = open('C:/Users/SEZE/Desktop/Master_thesis_2022/Data/C1_chromatograms/sample-set.json')
    # Read JSON
    data = json.load(file_path)

    #df['retention'] = round(df['retention'],3)



    # Number of wavelengths to test for
    n_wavelengths = len(data['items'][0]['chromatograms'])

    # List to contain all the old peak heights and retention times
    x_old_all = []
    y_old_all = []

    # Find all wavelengths and adjusted peak heights for each wave length and chromatogram
    for w in range(n_wavelengths):
        x_old = []
        y_old = []
        for r in range(len(data['items'])):
            retention            = data['items'][r]['chromatograms'][w]['x']
            injection_volume     = data['items'][r]['injection_volume']
            dilution             = data['items'][r]['dilution']
            peak_height          = data['items'][r]['chromatograms'][w]['y']
            adjusted_peak_height = [(p * injection_volume / dilution) for p in peak_height]
            x_old.append(retention)
            y_old.append(adjusted_peak_height)
        x_old_all.append(x_old)
        y_old_all.append(y_old)






    n_wavelengths = len(data['items'][0]['chromatograms'])
    wavelengths = []
    x_w_all = []
    y_w_all = []

    #df['retention_new'] = df['retention']
    df['retention_new'] = np.zeros(len(df))
    df['relative_area_new'] = np.zeros(len(df))

    c = 0
    for i in range(n_wavelengths):
        wl = data['items'][0]['chromatograms'][i]['channel_description'].split()[-1][:3]
        wavelengths.append(wl)


    # Get identifier to reconstruct spec_id
    #identifier = float(str(r) + '.' + wavelengths[w])

    for w in range(n_wavelengths):
        x_w = []
        y_w = []
        df_loop = list(set(df_peaks['spec_id'].loc[df_peaks['wave_length'].astype(str) == wavelengths[i]].astype(int)))
        #for r in range(len(data['items'])-1):
        #for r in range(len(df_loop)):
        for r in df_loop:
            #r += 1
            ids_ = df_loop[::-1]
            # Get identifier to reconstruct spec_id
            identifier = float(str(r) + '.' + wavelengths[w])
            #identifier = float(str(r) + '.' + wavelengths[w])
            
            if r != 0:
                for i in ids_:
                    id_to = i
                    
                    
                    # Conversion to un-warped
                    injection_volume     = data['items'][id_to]['injection_volume']
                    dilution             = data['items'][id_to]['dilution']
                    peak_height          = data['items'][id_to]['chromatograms'][w]['y']
                    adjusted_peak_height = [(p * injection_volume / dilution) for p in peak_height]
                    s2 = np.array(adjusted_peak_height)
                    # Get identifier to reconstruct spec_id
                    identifier = float(str(r) + '.' + wavelengths[w])
                    
                    if i == 1:
                        # Conversion from (in first case unwarped)
                        injection_volume     = data['items'][r]['injection_volume']
                        dilution             = data['items'][r]['dilution']
                        peak_height          = data['items'][r]['chromatograms'][w]['y']
                        adjusted_peak_height = [(p * injection_volume / dilution) for p in peak_height]
                        s1 = np.array(adjusted_peak_height)
                        
                        
                        ###########################################################
                        #####                map x and y values              ######
                        ###########################################################
                        
                        print(identifier)
                        # Find all peaks belonging to the spec_id
                        x_peaks_old = list(df['retention'].loc[df['spec_id'] == identifier])
                        x_peaks_old

                        # Is our s2
                        x_l = np.array(data['items'][r]['chromatograms'][w]['x'])

                        x_peaks_new = [0]*len(x_peaks_old)
                        x_peaks_id = []
                        # First peak values
                        y_peaks_new = [0]*len(x_peaks_old)

                        # Only when i == 0 (first time pr. chromatogram)
                        for x_r in x_peaks_old:
                            x_temp = min(x_l, key=lambda x:abs(x-x_r))
                            x_peaks_id.append([i for i, x in enumerate(x_l == x_temp) if x][0])
                        
                        # All other runs    
                        for x in range(len(x_peaks_id)):
                            id_from = np.array(path)[:,0]
                            id_to = np.array(path)[:,1]
                            idx_match = [i for i, x in enumerate(id_from == x_peaks_id[x]) if x][0]
                            x_peaks_id[x] = id_to[idx_match]
                            x_peaks_new[x] = x_l[id_to[idx_match]]
                            y_peaks_new[x] = s2[idx_match]
                        
                        
                        
                        
                    else:
                        
                        
                        # Calculate new y-values
                        #path = dtw.warping_path_fast(s1, s2, penalty = 0.05)
                        #path = dtw.warping_path_fast(s1, s2, penalty = 0.1)
                        path = dtw.warping_path_fast(s1, s2, penalty = 0.05, max_steps=(12))
                        ss = dtw.warp(s1,s2,path)
                        s1 = np.array(ss[0])
                        
                        ###########################################################
                        #####                map x and y values              ######
                        ###########################################################
                        
                        # All other runs    
                        for x in range(len(x_peaks_id)):
                            id_from = np.array(path)[:,0]
                            id_to = np.array(path)[:,1]
                            idx_match = [i for i, x in enumerate(id_from == x_peaks_id[x]) if x][0]
                            x_peaks_id[x] = id_to[idx_match]
                            x_peaks_new[x] = x_l[id_to[idx_match]]
                            y_peaks_new[x] = s2[idx_match]
                        
                        
                        
            else:
                s1 = np.array(data['items'][r]['chromatograms'][w]['y'])
                print(identifier+1)
                
            
            y_w.append(s1)
            idx_from_df = list(df['retention'].loc[df['spec_id'] == identifier].index)
            c+= 1 
            #print(c)
            for l in range(len(idx_from_df)):
                df['retention_new'][idx_from_df[l]] = x_peaks_new[l]
                df['relative_area_new'][idx_from_df[l]] = y_peaks_new[l]
            #print(identifier, ':', x_peaks_new)
            #print(identifier, ':', y_peaks_new)
            #print('w:', w, 'id:', r)
        y_w_all.append(y_w)
        #print(w)








    labels = ['Dimer', 'Main']
    t_214 = ['Chromatograms, with drift, before DTW - 214nm', 'Chromatograms, with drift, after DTW - 214nm']
    t_280 = ['Chromatograms, with drift, before DTW - 280nm', 'Chromatograms, with drift, after DTW - 280nm']
    w_id = [0,1]
    for i in set(w_id):
        
        fig, axs = plt.subplots(1, 2, figsize = (15,7))
        for j in range(len(t_214)):
            ax = format_axes(axs[j])
            for  p in range(len(y_old_all[i])):
                if j  == 0:
                    ax.plot(
                        x_old_all[i][p],
                        y_old_all[i][p],
                        #label = labels[i],
                        #cmap = mpl.colormaps['Paired']
                        )
                    ax.set_xlabel('Rt')
                    ax.set_ylabel('Adjusted peak height')
                else:
                    ax.plot(
                        x_old_all[i][p],
                        y_w_all[i][p],
                        #label = labels[i],
                        #cmap = mpl.colormaps['Paired']
                        )
                    ax.set_xlabel('Rt')
                    ax.set_ylabel('Adjusted peak height')
                if i == 0:
                    ax.set_title(t_214[j], pad = 20, fontsize = 18)
                else:
                    ax.set_title(t_280[j], pad = 20, fontsize = 18)
        plt.show()

    df = df['retention_new']
    df = df.rename(columns=({'retention_new':'retention'}))

    return df




def DTW_DRIFT_SIM(df, file_path):
    
    import json
    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    
    
    #plt.rcParams['figure.dpi'] = 600
    # Import file
    file_path = open('C:/Users/SEZE/Desktop/Master_thesis_2022/Data/C0_chromatograms/sample-set.json')
    #file_path = open('C:/Users/SEZE/Desktop/Master_thesis_2022/Data/C1_chromatograms/sample-set.json')
    # Read JSON
    data = json.load(file_path)
    
    #df['retention'] = round(df['retention'],3)
    
    
    n_wavelengths = len(data['items'][0]['chromatograms'])
    wavelengths = []
    x_w_all = []
    y_w_all = []
    
    #df_old_old = df
    #df = df_drift
    
    
    #df['retention_new'] = df['retention']
    df['retention_new'] = np.zeros(len(df))
    df['relative_area_new'] = np.zeros(len(df))
    
    c = 0
    for i in range(n_wavelengths):
        wl = data['items'][0]['chromatograms'][i]['channel_description'].split()[-1][:3]
        wavelengths.append(wl)
    
    
    
    
    # Number of wavelengths to test for
    n_wavelengths = len(data['items'][0]['chromatograms'])
    
    
    
    # List to contain all the old peak heights and retention times
    x_old_all = []
    y_old_all = []
    
    # Find all wavelengths and adjusted peak heights for each wave length and chromatogram
    for w in range(n_wavelengths):
        x_old = []
        y_old = []
        df_loop = list(set(df_peaks['spec_id'].loc[df_peaks['wave_length'].astype(str) == wavelengths[i]].astype(int)))
    
        for r in df_loop:
            retention            = data['items'][r]['chromatograms'][w]['x']
            injection_volume     = data['items'][r]['injection_volume']
            dilution             = data['items'][r]['dilution']
            peak_height          = data['items'][r]['chromatograms'][w]['y']
            adjusted_peak_height = [(p * injection_volume / dilution) for p in peak_height]
            adjusted_peak_height = np.append(np.array([0]*r*3), adjusted_peak_height[:(-r*3)]) # add minus section
            x_old.append(retention)
            y_old.append(adjusted_peak_height)
        x_old_all.append(x_old)
        y_old_all.append(y_old)
    
    
    
    
    
    
    
    # Get identifier to reconstruct spec_id
    #identifier = float(str(r) + '.' + wavelengths[w])
    
    for w in range(n_wavelengths):
        x_w = []
        y_w = []
        df_loop = list(set(df_peaks['spec_id'].loc[df_peaks['wave_length'].astype(str) == wavelengths[i]].astype(int)))
        #for r in range(len(data['items'])-1):
        #for r in range(len(df_loop)):
        for r in df_loop:
            #r += 1
            ids_ = df_loop[::-1]
            # Get identifier to reconstruct spec_id
            identifier = float(str(r) + '.' + wavelengths[w])
            #identifier = float(str(r) + '.' + wavelengths[w])
            
            if r != 0:
                for i in ids_:
                    id_to = i
                    
                    
                    # Conversion to un-warped
                    injection_volume     = data['items'][id_to]['injection_volume']
                    dilution             = data['items'][id_to]['dilution']
                    peak_height          = data['items'][id_to]['chromatograms'][w]['y']
                    adjusted_peak_height = [(p * injection_volume / dilution) for p in peak_height]
                    ## make and append statement with 
                    adjusted_peak_height = np.append(np.array([0]*i*3), adjusted_peak_height[:(-i*3)])
                    
                    s2 = np.array(adjusted_peak_height)
                    # Get identifier to reconstruct spec_id
                    identifier = float(str(r) + '.' + wavelengths[w])
                    
                    if i == 1:
                        # Conversion from (in first case unwarped)
                        injection_volume     = data['items'][r]['injection_volume']
                        dilution             = data['items'][r]['dilution']
                        peak_height          = data['items'][r]['chromatograms'][w]['y']
                        adjusted_peak_height = [(p * injection_volume / dilution) for p in peak_height]
                        adjusted_peak_height = np.append(np.array([0]*i*3), adjusted_peak_height[:(-i*3)])
                        s1 = np.array(adjusted_peak_height)
                        
                        
                        ###########################################################
                        #####                map x and y values              ######
                        ###########################################################
                        
                        print(identifier)
                        # Find all peaks belonging to the spec_id
                        x_peaks_old = list(df['retention'].loc[df['spec_id'] == identifier])
                        x_peaks_old
    
                        # Is our s2
                        x_l = np.array(data['items'][r]['chromatograms'][w]['x'])
    
                        x_peaks_new = [0]*len(x_peaks_old)
                        x_peaks_id = []
                        # First peak values
                        y_peaks_new = [0]*len(x_peaks_old)
    
                        # Only when i == 0 (first time pr. chromatogram)
                        for x_r in x_peaks_old:
                            x_temp = min(x_l, key=lambda x:abs(x-x_r))
                            x_peaks_id.append([i for i, x in enumerate(x_l == x_temp) if x][0])
                        
                        # All other runs    
                        for x in range(len(x_peaks_id)):
                            id_from = np.array(path)[:,0]
                            id_to = np.array(path)[:,1]
                            idx_match = [i for i, x in enumerate(id_from == x_peaks_id[x]) if x][0]
                            x_peaks_id[x] = id_to[idx_match]
                            x_peaks_new[x] = x_l[id_to[idx_match]]
                            y_peaks_new[x] = s2[idx_match]
                        
                        
                        
                        
                    else:
                        
                        
                        # Calculate new y-values
                        #path = dtw.warping_path_fast(s1, s2, penalty = 0.05)
                        #path = dtw.warping_path_fast(s1, s2, penalty = 0.1)
                        path = dtw.warping_path_fast(s1, s2, penalty = 0.05, max_steps=(12))
                        ss = dtw.warp(s1,s2,path)
                        s1 = np.array(ss[0])
                        
                        ###########################################################
                        #####                map x and y values              ######
                        ###########################################################
                        
                        # All other runs    
                        for x in range(len(x_peaks_id)):
                            id_from = np.array(path)[:,0]
                            id_to = np.array(path)[:,1]
                            idx_match = [i for i, x in enumerate(id_from == x_peaks_id[x]) if x][0]
                            x_peaks_id[x] = id_to[idx_match]
                            x_peaks_new[x] = x_l[id_to[idx_match]]
                            y_peaks_new[x] = s2[idx_match]
                        
                        
                        
            else:
                s1 = np.array(data['items'][r]['chromatograms'][w]['y'])
                print(identifier+1)
                
            
            y_w.append(s1)
            idx_from_df = list(df['retention'].loc[df['spec_id'] == identifier].index)
            c+= 1 
            #print(c)
            for l in range(len(idx_from_df)):
                df['retention_new'][idx_from_df[l]] = x_peaks_new[l]
                df['relative_area_new'][idx_from_df[l]] = y_peaks_new[l]
            #print(identifier, ':', x_peaks_new)
            #print(identifier, ':', y_peaks_new)
            #print('w:', w, 'id:', r)
        y_w_all.append(y_w)
        #print(w)
    
    
    
    
    
    
    
    labels = ['Dimer', 'Main']
    t_214 = ['Chromatograms, with drift, before DTW - 214nm', 'Chromatograms, with drift, after DTW - 214nm']
    t_280 = ['Chromatograms, with drift, before DTW - 280nm', 'Chromatograms, with drift, after DTW - 280nm']
    w_id = [0,1]
    for i in set(w_id):
        
        fig, axs = plt.subplots(1, 2, figsize = (15,7))
        for j in range(len(t_214)):
            ax = format_axes(axs[j])
            for  p in range(len(y_old_all[i])):
                if j  == 0:
                    ax.plot(
                        x_old_all[i][p],
                        y_old_all[i][p],
                        #label = labels[i],
                        #cmap = mpl.colormaps['Paired']
                        )
                    ax.set_xlabel('Rt')
                    ax.set_ylabel('Adjusted peak height')
                else:
                    ax.plot(
                        x_old_all[i][p],
                        y_w_all[i][p],
                        #label = labels[i],
                        #cmap = mpl.colormaps['Paired']
                        )
                    ax.set_xlabel('Rt')
                    ax.set_ylabel('Adjusted peak height')
                if i == 0:
                    ax.set_title(t_214[j], pad = 20, fontsize = 18)
                else:
                    ax.set_title(t_280[j], pad = 20, fontsize = 18)
        plt.show()
        
    df = df['retention_new']
    df = df.rename(columns=({'retention_new':'retention'}))
     
    return df

