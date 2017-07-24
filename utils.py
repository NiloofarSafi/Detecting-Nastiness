import os


def read_results(fname):
    found = 0
    with open(fname, 'r') as f_in:
        accuracy = 0
        for line in f_in:
            line = line.strip()
            if line.startswith("Feature"):

                _, feature_name = line.split()

            elif line.startswith('negative'):
                vals_n = line.split()
                negative_p, negative_r, negative_f = vals_n[1], vals_n[2], vals_n[3]

            elif line.startswith('positive'):
                vals_neutral = line.split()
                neutral_p, neutral_r, neutral_f = vals_neutral[1], vals_neutral[2], vals_neutral[3]

            elif line.startswith("Test Accuracy:"):
                _, _,accuracy = line.split()

            elif line.startswith('Macro Precision Score'):
                vals_p = line.split(',')
                precision_macro, precision_micro, precision_weighted = vals_p[1], vals_p[3], vals_p[5]
            elif line.startswith('Macro Recall score'):
                vals_r = line.split(',')
                recall_macro, recall_micro, recall_weighted = vals_r[1], vals_r[3], vals_r[5]

            elif line.startswith('Macro F1-score'):
                vals_f = line.split(',')
                f_macro, f_micro, f_weighted = vals_f[1], vals_f[3], vals_f[5]
            elif line.startswith('ROC AUC'):
                _, _,auc = line.split()

    return feature_name, accuracy, precision_macro, recall_macro, f_macro, precision_micro, recall_micro, f_micro, precision_weighted, recall_weighted, f_weighted, negative_p, negative_r, negative_f, neutral_p, neutral_r, neutral_f, auc


def display_classification_results(result_dir):
    for file in os.listdir(result_dir):
        if os.path.isfile(os.path.join(result_dir, file)):
            #print(file)
            feature_name, accuracy, precision, recall, fscore, precision_m, recall_m, fscore_m, precision_w, recall_w, fscore_w, negative_p, negative_r, negative_f, neutral_p, neutral_r, neutral_f, auc = read_results(
                os.path.join(result_dir, file))
            print (
                '{feature},{auc},{accuracy},{negative_precision},{negative_recall},{negative_fscore},{positive_precision},{positive_recall},{positive_fscore},{precision},{recall},{fscore},{precision_m},{recall_m},{fscore_m},{precision_w},{recall_w},{fscore_w}'.format(
                    feature=feature_name, auc=auc, accuracy=accuracy,

                    negative_precision=negative_p, negative_recall=negative_r, negative_fscore=negative_f,
                    positive_precision=neutral_p, positive_recall=neutral_r, positive_fscore=neutral_f,
                    precision=precision, recall=recall,
                    fscore=fscore, precision_m=precision_m, recall_m=recall_m,
                    fscore_m=fscore_m, precision_w=precision_w, recall_w=recall_w,
                    fscore_w=fscore_w))
