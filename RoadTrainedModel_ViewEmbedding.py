import numpy as np

from visualizer import VisualResult

from ProjectionCS import ShowComplementarySubspace, ShowRectangularCoordinateSystem

if __name__ == '__main__':
    data = np.load('datastore\Trained_Data\history_20260215_174350.npy', allow_pickle=True).item()
    X =np.array(data['X'])
    VisualResult(data['U'][-1, :, :], data['V'][-1, :, :], data['E'], None, None, None, None, None, None, label1=data['object'], label2=data['tag'])

    if data['U'][-1,-1,:].size == 3 and data['V'][-1,-1,:].size == 3:
        colors = np.array(['k'] * len(X[-1]),dtype=object)
        # 正射影と余射影の両方を表示する関数
        ShowComplementarySubspace(data['U'][-1, :, :], data['V'][-1, :,:], colors, X, data['object'], data['tag']) #1つの視点で垂直射影するならこれ
        # ユーザが選択した2つのview pointを直交座標系で可視化する関数
        ShowRectangularCoordinateSystem(X, data['U'][-1, :, :], data['V'][-1, :, :], data['object'], data['tag'])