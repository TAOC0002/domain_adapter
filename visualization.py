from sklearn.manifold import TSNE
import pandas as pd
import torch
import os
import seaborn as sns
import numpy as np
import matplotlib as mplt
import matplotlib.pylab as plt
import pickle
import itertools
import sys

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

# plot and save tsne
def plot_tsne(df_tsne, classes, markers_class, epoch, mode, save_path=None, save_prefix='tsne', apply_legend=False):
    if save_path is None:
        print('Specify save_path.')
        sys.exit()
    else:
        if not(os.path.exists(save_path)):
            os.makedirs(save_path)
        figure_size= (12,8)
        font_size=32
        mplt.rcParams.update({'font.size': font_size})
        # color by class
        fig = plt.figure()
        fig.set_size_inches( figure_size )
        fname2 = os.path.join(save_path, save_prefix+'epoch'+str(epoch)+'-'+mode+'.png')
#         palette = sns.hls_palette(len(np.unique(df_tsne['Class'])), l=0.3, s=0.8)
        palette = sns.hls_palette(len(np.unique(df_tsne['Class'])))
        plt2 = sns.scatterplot(
                x="tsne-axis1", y="tsne-axis2",
                hue="Class",
                style="Class",
                palette={i: palette[i] for i,c in enumerate(classes)},
                markers={i: markers_class[i] for i,c in enumerate(classes)},
                data=df_tsne,
                legend="full",
                # alpha=0.3
            )
        box = plt2.get_position()
        plt2.set_position([box.x0, box.y0 + box.height * 0.1,
            box.width, box.height * 0.8])
        handles, labels = plt2.get_legend_handles_labels()
        order = [[i for i,l in enumerate(labels) if l==c][0] for c in classes]
        if apply_legend:
            plt.legend(flip([handles[i] for i in order], 5), flip([labels[i] for i in order], 5),
                loc='upper center', bbox_to_anchor=(0.5,-0.05), borderaxespad=0, columnspacing=0, labelspacing=0,
                fontsize=font_size, markerscale=3, handletextpad=0, borderpad=0, ncol=6, frameon=False)   
        else:
            plt2.get_legend().remove()
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])        
        plt2.axis("off")
        plt2.figure.savefig(fname2)


tsne_feats_dir = '/home/taochen/meta-learning/DomainAdaptor/t-sne/'
tsne_outputs_dir = '/home/taochen/meta-learning/DomainAdaptor/t-sne-outputs/'
classes = [str(i) for i in range(5)]
markers_class = ['o']*5
feats, y = None, None

# # global epoch visualization
# for epoch in [0, 1, 2]:
#     for mode in ['eval', 'test']:
#         for step in range(865):  # 1131
#             if step == 0:
#                 feats = torch.load("{}epoch{}-step{}-mode_{}-feats.pt".format(tsne_feats_dir, epoch, step, mode))
#                 y = torch.load("{}epoch{}-step{}-mode_{}-labels.pt".format(tsne_feats_dir, epoch, step, mode))
#             else:
#                 feats = torch.cat((feats, torch.load("{}epoch{}-step{}-mode_{}-feats.pt".format(tsne_feats_dir, epoch, step, mode))), 0)
#                 y = torch.cat((y, torch.load("{}epoch{}-step{}-mode_{}-labels.pt".format(tsne_feats_dir, epoch, step, mode))), 0)
            
#         df = pd.DataFrame()
#         tsne = TSNE(n_components=2, verbose=1, random_state=123)
#         z = tsne.fit_transform(feats.detach().cpu().numpy())
#         df["Class"] = y.detach().cpu().numpy()
#         df["tsne-axis1"] = z[:,0]
#         df["tsne-axis2"] = z[:,1]
#         plot_tsne(df, classes, markers_class, epoch, mode, save_path=tsne_outputs_dir+'visda17-global', save_prefix='')

# # local MME effect check
# for epoch in [0, 1, 2, 9]:
#     for mode in ['eval', 'test']:
#         for step in ['200', '400', '600']:
#             for stage in ['orig', 'max', 'min']:
#                 feats = torch.load("{}epoch{}-step{}-mode_{}-{}-feats.pt".format(tsne_feats_dir, epoch, step, mode, stage))
#                 y = torch.load("{}epoch{}-step{}-mode_{}-{}-labels.pt".format(tsne_feats_dir, epoch, step, mode, stage))
#                 df = pd.DataFrame()
#                 tsne = TSNE(n_components=2, verbose=1, random_state=123)
#                 z = tsne.fit_transform(feats.detach().cpu().numpy())
#                 df["Class"] = y.detach().cpu().numpy()
#                 df["tsne-axis1"] = z[:,0]
#                 df["tsne-axis2"] = z[:,1]
#                 try:
#                     plot_tsne(df, classes, markers_class, epoch, mode, save_path=tsne_outputs_dir+'visda17-local', save_prefix=stage+'-step'+step+'_')
#                 except:
#                     pass


# global epoch visualization
for epoch in [0]:
    for mode in ['eval', 'test']:
        for stage in ['orig', 'max', 'min']:
            for step in range(11):  # 865, 1131
                if step == 0:
                    feats = torch.load("{}epoch{}-step{}-mode_{}-{}-feats.pt".format(tsne_feats_dir, epoch, step, mode, stage))
                    y = torch.load("{}epoch{}-step{}-mode_{}-{}-labels.pt".format(tsne_feats_dir, epoch, step, mode, stage))
                else:
                    feats = torch.cat((feats, torch.load("{}epoch{}-step{}-mode_{}-{}-feats.pt".format(tsne_feats_dir, epoch, step, mode, stage))), 0)
                    y = torch.cat((y, torch.load("{}epoch{}-step{}-mode_{}-{}-labels.pt".format(tsne_feats_dir, epoch, step, mode, stage))), 0)
                
            df = pd.DataFrame()
            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(feats.detach().cpu().numpy())
            df["Class"] = y.detach().cpu().numpy()
            df["tsne-axis1"] = z[:,0]
            df["tsne-axis2"] = z[:,1]
            plot_tsne(df, classes, markers_class, epoch, mode, save_path=tsne_outputs_dir+'vlcs-local', save_prefix='11-'+stage+'_')


