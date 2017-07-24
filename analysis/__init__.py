import pandas as pd


def show_mistakes(fname, X, y_act, y_pred):
    mistake_ind = [y_act != y_pred]
    mistakes = zip(X[mistake_ind], y_act[mistake_ind], y_pred[mistake_ind])

    data = []
    for mistake, act, pred in mistakes:
        row = {'id': mistake.id, 'content': mistake.content, 'type': mistake.post_type, 'actual': "Positive" if act else 'Negative',
               'predicted': "Positive" if pred else 'Negative'}
        data.append(row)

    df = pd.DataFrame(data,columns=['id','type','content','actual','predicted'])
    df.to_csv(fname, encoding='utf=8',index=False)
