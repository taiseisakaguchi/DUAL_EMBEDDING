import os
import pandas as pd

def load_data(retlabel_object, retlabel_tag):
    # CSVファイルが保存されているディレクトリのパス
    file_path = '.\datastore\Kohonen_Animals\Kohonen_animals.csv'
    # CSVファイルの読み込み
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # 前後の空白を削除

    X = df.drop(df.columns[0], axis='columns').to_numpy()
    return_data = [X]

    if retlabel_object:
        name_U = df.iloc[:,0].to_numpy()
        return_data.append(name_U)

    if retlabel_tag:
        name_V = df.columns[1:].to_numpy()
        return_data.append(name_V)
    return return_data

if __name__ == "__main__":  # このファイルを実行した時のみ
    data = load_data(retlabel_object=True, retlabel_tag=True)
    print('学習データの形:{}'.format(data[0].shape))
    print('アイテムの数:{}'.format(data[1].shape))
    print('タグの数:{}'.format(data[2].shape))