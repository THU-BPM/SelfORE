import json
import pandas as pd
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix

def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes

def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.
    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.
    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.
    sys_labels : ndarray, (n_frames,)
        System labels.
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)
    Returns
    -------
    precision : float
        B-cubed precision.
    recall : float
        B-cubed recall.
    f1 : float
        B-cubed F1.
    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    ref_labels = np.array(ref_labels)
    sys_labels = np.array(sys_labels)
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1

def cls_report(K_NUM, LOOP, DATASET="nyt_fb_v2", LABEL_NUM = 10):
    # LABEL_NUM is related to DATASET (10 or 15)
#     DATASET = "nyt_fb_v2"
    train_data=[]
    label=[]

    k_means_label_json_file = DATASET + "_" + str(K_NUM) + "_fixed_layer_k_means_label_" + str(LOOP) + ".json"
    idx_json_file = DATASET + "_" + str(K_NUM) +  "_fixed_layer_label_ids_" + str(LOOP) + ".json" 
    train_data=json.load(open(k_means_label_json_file, 'r'))
    m=json.load(open(idx_json_file, 'r'))
    # import ipdb; ipdb.set_trace()
    for i in range(len(m)):
        for j in range(len(m[i])):
            label.append(m[i][j])
    result=[]
    number=[]
    pesudo_true_label=[]
    name=[]
    for i in range(LABEL_NUM):
        name.append(i)
    for i in range(len(train_data)):
        pesudo_true_label.append(0)
    for i in range(K_NUM):
        number.append(0)
    for j in range(K_NUM):
        temper_2 = []
        temper=0
        for i in range(LABEL_NUM):
            temper_2.append(0)
        for i in range(len(train_data)):
            if(train_data[i]==j):
                temper_2[label[i]]+=1
                temper=temper+1
        number[j]=temper
        sum=0
        max=0
        num=0
        for i in range(len(temper_2)):
            sum=sum+temper_2[i]
            if(temper_2[i]>=max):
                max=temper_2[i]
                num=i
        for i in range(len(train_data)):
            if(train_data[i]==j):
                pesudo_true_label[i]=num
        if(sum!=0):
            result_=max/sum
            result.append(result_)
        if(sum==0):
            result.append(0)
    total_1=0
    for i in range(LABEL_NUM):
        total_1=total_1+result[i]*number[i]

    from sklearn.metrics.cluster import homogeneity_completeness_v_measure
    from sklearn.metrics import classification_report
    from sklearn.metrics.cluster import adjusted_rand_score
    
    return pesudo_true_label, label
    print(len(label), len(pesudo_true_label))
    
    print(classification_report(label, pesudo_true_label,labels=name))


def usoon_eval(DATASET, K_NUM, LABEL_NUM, LOOP, NAME=False):
    
    train_data=[]
    label=[]

    k_means_label_json_file = DATASET + "_" + str(K_NUM) + "_fixed_layer_k_means_label_" + str(LOOP) + ".json"
    idx_json_file = DATASET + "_" + str(K_NUM) +  "_fixed_layer_label_ids_" + str(LOOP) + ".json" 
    train_data=json.load(open(k_means_label_json_file, 'r'))
    m=json.load(open(idx_json_file, 'r'))
    #import ipdb; ipdb.set_trace()
    for i in range(len(m)):
        for j in range(len(m[i])):
            label.append(m[i][j])
    result=[]
    number=[]
    pesudo_true_label=[]
    name=[]
    for i in range(LABEL_NUM):
        name.append(i)
    for i in range(len(train_data)):
        pesudo_true_label.append(0)
    for i in range(K_NUM):
        number.append(0)
    for j in range(K_NUM):
        temper_2 = []
        temper=0
        for i in range(LABEL_NUM):
            temper_2.append(0)
        for i in range(len(train_data)):
            if(train_data[i]==j):
                temper_2[label[i]]+=1
                temper=temper+1
        number[j]=temper
        sum=0
        max=0
        num=0
        for i in range(len(temper_2)):
            sum=sum+temper_2[i]
            if(temper_2[i]>=max):
                max=temper_2[i]
                num=i
        for i in range(len(train_data)):
            if(train_data[i]==j):
                pesudo_true_label[i]=num
        if(sum!=0):
            result_=max/sum
            result.append(result_)
        if(sum==0):
            result.append(0)
    total_1=0
    for i in range(LABEL_NUM):
        total_1=total_1+result[i]*number[i]

    from sklearn.metrics.cluster import homogeneity_completeness_v_measure
    from sklearn.metrics import classification_report
    from sklearn.metrics.cluster import adjusted_rand_score
    
    ARI = adjusted_rand_score(label, pesudo_true_label)
    if NAME == True:
        return label, pesudo_true_label # test
    
    # res_dic = classification_report(label, pesudo_true_label,labels=name, output_dict=True)
    # return precision, recall, f1
    B3_prec, B3_rec, B3_f1 = bcubed(label, pesudo_true_label)
    # B3_f1 = res_dic["weighted avg"]['f1-score']
    # B3_prec = res_dic["weighted avg"]['precision']
    # B3_rec = res_dic["weighted avg"]['recall']
    
    v_hom, v_comp, v_f1 = homogeneity_completeness_v_measure(label, pesudo_true_label)
    return B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI

def get_df(DATASET, K_NUM, LABEL_NUM, LOOP_NUM):
    import pandas as pd
    line = []
    table = []

    for loop in range(LOOP_NUM):
        B1_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI =  usoon_eval(DATASET, K_NUM, LABEL_NUM, loop)
        line = [B1_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI]
        line = [l*100 for l in line]

        table.append(line)
    df = pd.DataFrame(table)
    df.columns = ['B3_F1', 'B3_Prec.', 'B3_Rec.', 'V_F1', 'V_Hom.', 'V_Comp.', 'ARI']
    return df

def pretty_df(df, FLOAT, DATASET, K_NUM):
    from tabulate import tabulate
    print("K: {} \t Dataset: {} ".format(K_NUM, DATASET))
    
    float_fmt = "." + str(FLOAT) + "f"
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=float_fmt, numalign='center'))
        

def measure(K_NUMS, LABEL_NUM, DATASET, LOOP_NUM, FLOAT=1):
    for K_NUM in K_NUMS:
        df = get_df(DATASET, K_NUM, LABEL_NUM, LOOP_NUM)
        pretty_df(df, FLOAT, DATASET, K_NUM)   
