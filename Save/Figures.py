import numpy as np
import matplotlib.pyplot as plt
import os

def save_pca_figure(explained_variance_ratio_, n_component, feature_name, sub_idx, date, savepath=None):
    xlabel = []
    Ratio = explained_variance_ratio_ * 100
    cumulative_Ratio = np.cumsum(Ratio)
    Ratio = np.append(Ratio, 100 - cumulative_Ratio[-1])
    cumulative_Ratio = np.append(cumulative_Ratio, 100.0)

    for i in range(1, n_component + 1):
        xlabel.append(str(i))
    xlabel.append('(' + str(n_component + 1) + '~ 310)')

    plt.clf()
    plt.bar(xlabel, Ratio, color='blue')
    plt.plot(xlabel, cumulative_Ratio, visible=True, marker='*', linestyle='-', color='red')
    plt.title("PCA results of {} feature samples".format(feature_name))
    plt.xlabel("N_components")
    plt.ylabel("Explaned variance ratio (%)")
    plt.xticks(range(n_component + 1), xlabel, fontsize=5)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylim([0, 101])
    for i in range(n_component + 1):
        height = cumulative_Ratio[i]
        if i > 0:
            plt.text(xlabel[i], height - 8, '%.1f' % height, ha='center', va='bottom', size=8)
        height2 = Ratio[i]
        plt.text(xlabel[i], height2, '%.1f' % height2, ha='center', va='bottom', size=8, color='black')

    plt.legend(['CDF', 'PDF'])
    # plt.show()
    if savepath != None:
        os.chdir(savepath)
    plt.savefig('./store/figure/pca_result/' + sub_idx + '/' + feature_name + '/' + feature_name + '_' + date + '.png',
                dpi=250, facecolor='#eeeeee')

    return


def save_heatmap(matrix, title, xlabel, ylabel, save_path, clim_min=None, clim_max=None, _dpi=300, _facecolor="#eeeeee",
                 _bbox_inches='tight'):
    plt.clf()
    plt.matshow(matrix)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    if clim_min != None and clim_max != None:
        plt.clim(clim_min, clim_max)
    plt.savefig(save_path, dpi=_dpi, facecolor=_facecolor, bbox_inches=_bbox_inches)

    return


def save_pca_heatmap(feature, sample_counts, feature_name, n_sessions, n_trials, sub_idx, date, _save_path):
    label_list = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    s = 0
    e = 0
    idx = 0

    pca_min = feature.min()
    pca_max = feature.max()
    for i in range(n_sessions):
        for j in range(n_trials):
            idx = n_trials * i + j
            e += sample_counts[idx]
            if label_list[j] == 1:
                save_path = _save_path + 'heatmap/' + sub_idx + '/' + feature_name + '/positive/' + feature_name + '_positive_idx' + str(
                    idx) + '_' + date + '.png'
                save_heatmap(de_pca[s:e], title='PCA_results - ' + feature_name + '_positive',
                             xlabel="feature dimension",
                             ylabel="sample counts", clim_min=pca_min, clim_max=pca_max, save_path=save_path)
            elif label_list[j] == 0:
                save_path = _save_path + 'heatmap/' + sub_idx + '/' + feature_name + '/neutral/' + feature_name + '_neutral_idx' + str(
                    idx) + '_' + date + '.png'
                save_heatmap(de_pca[s:e], title='PCA_results - ' + feature_name + '_netural',
                             xlabel="feature dimension",
                             ylabel="sample counts", clim_min=pca_min, clim_max=pca_max, save_path=save_path)
            elif label_list[j] == -1:
                save_path = _save_path + 'heatmap/' + sub_idx + '/' + feature_name + '/negative/' + feature_name + '_negative_idx' + str(
                    idx) + '_' + date + '.png'
                save_heatmap(de_pca[s:e], title='PCA_results - ' + feature_name + '_negative',
                             xlabel="feature dimension",
                             ylabel="sample counts", clim_min=pca_min, clim_max=pca_max, save_path=save_path)
            else:
                print("Error - Invalid label number : ", label_list[j])

            s = e
    return


def save_scatter(result, label, best_acc, sub_idx, date, save_path, dr_type, iden):
    negw = np.where(label.cpu() == 0)[0]
    neuw = np.where(label.cpu() == 1)[0]
    posw = np.where(label.cpu() == 2)[0]
    if dr_type == 0:
        dr_name = 'PCA'
    elif dr_type == 1:
        dr_name = 'TSNE'
    else:
        dr_name = ''
    neg = result[negw]
    neu = result[neuw]
    pos = result[posw]

    gts = np.where(iden.cpu().numpy() == True)[0]

    neggtw, neugtw, posgtw = [], [], []

    for i in gts:
        if i in negw:
            neggtw.append(i)
        elif i in neuw:
            neugtw.append(i)
        elif i in posw:
            posgtw.append(i)

    neggt = result[ngtw, :]
    neugt = result[sgtw, :]
    posgt = result[fgtw, :]

    plt.clf()
    c1 = plt.scatter(neg[:, 0], neg[:, 1], marker="o", color='#FF6600', s=1.)
    gt1 = plt.scatter(neggt[:, 0], neggt[:, 1], marker="x", color='red', s=2.)
    c2 = plt.scatter(neu[:, 0], neu[:, 1], marker="o", color='#CCFF66', s=1.)
    gt2 = plt.scatter(neugt[:, 0], neugt[:, 1], marker="x", color='green', s=2.)
    c3 = plt.scatter(pos[:, 0], pos[:, 1], marker="o", color='#33CCFF', s=1.)
    gt3 = plt.scatter(posgt[:, 0], posgt[:, 1], marker="x", color='blue', s=2.)

    plt.title('Classfication result of ' + sub_idx + ' - Acc : ' + str(best_acc) + '%')
    plt.xlabel("Reduced axis - 1")
    plt.ylabel("Reduced axis - 2")
    plt.legend(handles=(c1, c2, c3, gt1, gt2, gt3),
               labels=("negative", "neutral", "positive", "negative-gt", "neutral-gt", "positive-gt"))
    plt.savefig(
        save_path + 'test_result/' + sub_idx + '/' + dr_name + '_classification_result_scatter_' + date + '.png',
        dpi=300, facecolor="#eeeeee", bbox_inches='tight')
    return


def save_scatter_seedIV(result, label, best_acc, sub_idx, date, save_path, dr_type, iden):
    nw = np.where(label.cpu() == 0)[0]
    sw = np.where(label.cpu() == 1)[0]
    fw = np.where(label.cpu() == 2)[0]
    hw = np.where(label.cpu() == 3)[0]
    if dr_type == 0:
        dr_name = 'PCA'
    elif dr_type == 1:
        dr_name = 'TSNE'
    else:
        dr_name = ''

    n = result[nw, :]
    s = result[sw, :]
    f = result[fw, :]
    h = result[hw, :]

    gts = np.where(iden.cpu().numpy() == True)[0]

    ngtw, sgtw, fgtw, hgtw = [], [], [], []
    for i in gts:
        if i in nw:
            ngtw.append(i)
        elif i in sw:
            sgtw.append(i)
        elif i in fw:
            fgtw.append(i)
        else:
            hgtw.append(i)

    ngt = result[ngtw, :]
    sgt = result[sgtw, :]
    fgt = result[fgtw, :]
    hgt = result[hgtw, :]

    plt.clf()
    c1 = plt.scatter(n[:, 0], n[:, 1], marker="o", color='#999999', s=1.)
    gt1 = plt.scatter(ngt[:, 0], ngt[:, 1], marker="x", color='black', s=2.)
    c2 = plt.scatter(s[:, 0], s[:, 1], marker="o", color='#CCFF66', s=1.)
    gt2 = plt.scatter(sgt[:, 0], sgt[:, 1], marker="x", color='green', s=2.)
    c3 = plt.scatter(f[:, 0], f[:, 1], marker="o", color='#FF6600', s=1.)
    gt3 = plt.scatter(fgt[:, 0], fgt[:, 1], marker="x", color='red', s=2.)
    c4 = plt.scatter(h[:, 0], h[:, 1], marker="o", color='#33CCFF', s=1.)
    gt4 = plt.scatter(hgt[:, 0], hgt[:, 1], marker="x", color='blue', s=2.)
    plt.title('Classfication result of ' + sub_idx + ' - Acc : ' + str(best_acc) + '%')
    plt.xlabel("Reduced axis - 1")
    plt.ylabel("Reduced axis - 2")
    plt.legend(handles=(c1, c2, c3, c4, gt1, gt2, gt3, gt4),
               labels=("neutral", "sad", "fear", "happy", "neutral - gt", "sad - gt", "fear - gt", "happy - gt"))
    plt.savefig(
        save_path + 'test_result/' + sub_idx + '/' + dr_name + '_classification_result_scatter_' + date + '.png',
        dpi=300, facecolor="#eeeeee", bbox_inches='tight')
    return


def save_scatter_deap(result, label, best_acc, sub_idx, date, save_path, dr_type, iden, cf_type, fig_name):
    if dr_type == 0:
        dr_name = 'PCA'
    elif dr_type == 1:
        dr_name = 'TSNE'
    else:
        dr_name = ''

    lw = np.where(label.cpu() == 0)[0]
    hw = np.where(label.cpu() == 1)[0]
    l = result[lw, :]
    h = result[hw, :]

    gts = np.where(iden.cpu().numpy() == True)[0]

    lgtw, hgtw = [], []

    for i in gts:
        if i in lw:
            lgtw.append(i)
        else:
            hgtw.append(i)

    lgt = result[lgtw, :]
    hgt = result[hgtw, :]

    plt.clf()
    c1 = plt.scatter(l[:, 0], l[:, 1], marker="o", color='#FF6600', s=1.)
    gt1 = plt.scatter(lgt[:, 0], lgt[:, 1], marker="x", color='red', s=2.)
    c2 = plt.scatter(h[:, 0], h[:, 1], marker="o", color='#33CCFF', s=1.)
    gt2 = plt.scatter(hgt[:, 0], hgt[:, 1], marker="x", color='blue', s=2.)
    plt.title('Classfication result of ' + sub_idx + ' - Acc : ' + str(best_acc) + '%')
    plt.xlabel("Reduced axis - 1")
    plt.ylabel("Reduced axis - 2")
    plt.legend(handles=(c1, c2, gt1, gt2),
               labels=("Low " + cf_type, "High " + cf_type, "Low " + cf_type + " - gt", "High " + cf_type + " - gt"))
    plt.savefig(
        save_path + 'test_result/' + sub_idx + '/' + fig_name + '_' + dr_name + '_classification_result_scatter_' + date + '.png',
        dpi=300, facecolor="#eeeeee", bbox_inches='tight')
    return