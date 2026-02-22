import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as patheffects
import japanize_matplotlib
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.backend_bases import MouseEvent
from adjustText import adjust_text


def Projection1D(ax, X, ax_data, view_vec, color,label):
    ax.tick_params(labelbottom=True, bottom=False) #x軸設定
    ax.tick_params(labelleft=False, left=False) #y軸設定
    d = np.sum(ax_data * view_vec, axis=1)/np.linalg.norm(view_vec)
    for i, x in enumerate(X):
        if x!=1:
            ax.scatter(d[i], 0.0, color = 'k')
        else :
            ax.scatter(d[i],0.0,color=color)
    texts = [ax.text(d[i],0.0, label[i], fontsize=12,path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")]) for i in range(len(X))]
    adjust_text(texts)

def plot_line(ax_line, ax_point, index, X, line_space, point_space, sub_vec, color):
    ax_point.scatter(point_space[index,0],point_space[index,1], color = color)
    # 単位ベクトルの2乗
    sub_vec = sub_vec/np.linalg.norm(sub_vec)**2
    # 直線を描画
    ax_line.axline((0, (sub_vec[0]**2 + sub_vec[1]**2)/sub_vec[1]), slope = -(sub_vec[0] / sub_vec[1]), color = color)
    for i ,x in enumerate(X):
        if x !=0:
            ax_line.scatter(line_space[i,0], line_space[i,1], color = color)
    ax_line.axhline(0, color='black',linewidth=0.5)
    ax_line.axvline(0, color='black',linewidth=0.5)
    xlim = ax_line.get_xlim()
    ylim = ax_line.get_ylim()
    lim = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    ax_line.set_xlim(-lim, lim)
    ax_line.set_ylim(-lim, lim)
    ax_line.set(aspect='equal')

def draw_latent_2D(ax, Z, label):
    ax.scatter(Z[:, 0], Z[:, 1], color = 'k')
    if label is not None:
        for i, name in enumerate(label): #全ベクトルの名前を可視化したいならこっち
            ax.text(Z[i, 0], Z[i, 1], name, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
    else:
        for i in range(len(Z)): #全ベクトルの番号を可視化
            ax.text(Z[i, 0], Z[i, 1], i, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    ax.set_xlim(-(lim + 0.05), lim + 0.05)
    ax.set_ylim(-(lim + 0.05), lim + 0.05)
    # ax.set(aspect='equal')
    # ax.legend(bbox_to_anchor=(0, -0.1), loc='upper left_', borderaxespad=0, fontsize=18)

def show_name(ax, X,label):
    text =[[] for i in range(int(np.sum(X)))]
    j=0
    for i , x in enumerate(X):
        if x !=0:
            text[j].append(i)
            text[j].append(label[i])
            j+=1
    ax.table(cellText=text, colLabels=['固有ID', 'ラベル名'], cellLoc='center',loc='center', colColours=['#808080', '#808080', '#808080'])


def on_click(event):
    """
    クリックイベントに応じてインデックスを取得し、直線を描画する。

    Parameters:
    event : MouseEvent
        マウスクリックイベント。
    """
    global Visual_object_index, Visual_tag_index  # 動的に更新するインデックス

    if event.inaxes == latent1_ax:  # U空間がクリックされた場合
        click_x, click_y = event.xdata, event.ydata
        distances = np.sqrt((U[:, 0] - click_x)**2 + (U[:, 1] - click_y)**2)
        Visual_object_index = np.argmin(distances)  # 最近傍のインデックスを取得
        print(f"U空間のインデックス {Visual_object_index} が選択されました")

        latent1_ax.clear()
        latent2_ax.clear()
        latent3_ax.clear()
        latent3_ax.set_axis_off()
        latent4_ax.clear()
        draw_latent_2D(latent1_ax, U, label1)
        latent1_ax.text(U[Visual_object_index,0], U[Visual_object_index,1], label1[Visual_object_index], path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
        # 直線を再描画
        draw_latent_2D(latent2_ax, V, label2)
        plot_line(latent2_ax,latent1_ax, Visual_object_index, X[Visual_object_index,:], V, U, U[Visual_object_index], colors_item[Visual_object_index])
        Projection1D(latent4_ax, X[Visual_object_index,:], V, U[Visual_object_index], colors_item[Visual_object_index],label2)
        fig1.canvas.draw()
        fig2.canvas.draw()


    elif event.inaxes == latent2_ax:  # V空間がクリックされた場合
        click_x, click_y = event.xdata, event.ydata
        distances = np.sqrt((V[:, 0] - click_x)**2 + (V[:, 1] - click_y)**2)
        Visual_tag_index = np.argmin(distances)  # 最近傍のインデックスを取得
        print(f"V空間のインデックス {Visual_tag_index} が選択されました")

        latent2_ax.clear()
        draw_latent_2D(latent2_ax, V, label2)
        # 直線を再描画
        latent1_ax.clear()
        latent3_ax.clear()
        latent4_ax.clear()
        draw_latent_2D(latent1_ax, U, label1)
        plot_line(latent1_ax,latent2_ax, Visual_tag_index, X[:, Visual_tag_index], U, V, V[Visual_tag_index], colors_tag[Visual_tag_index])
        Projection1D(latent4_ax, X[:,Visual_tag_index], U, V[Visual_tag_index], colors_tag[Visual_tag_index],label1)
        # show_name(latent3_ax, X[:, Visual_tag_index],label1)
        latent3_ax.set_axis_off()
        fig1.canvas.draw()
        fig2.canvas.draw()

    elif event.inaxes == latent4_ax:  # 数直線がクリックされた場合
        pass

    else:
        latent1_ax.clear()
        latent2_ax.clear()
        latent3_ax.clear()
        latent3_ax.set_axis_off()
        latent4_ax.clear()
        draw_latent_2D(latent1_ax, U, label1)
        draw_latent_2D(latent2_ax, V, label2)
        fig1.canvas.draw()
        fig2.canvas.draw()

# グローバル変数としてインデックスを初期化
Visual_object_index = None
Visual_tag_index = None

def Click_and_View_1D(X_input, U_input, V_input, colors_item_input, colors_tag_input,label1_input=None, label2_input=None):
    global fig1, fig2, latent1_ax, latent2_ax,latent3_ax, latent4_ax, X, U, V, label1, label2, colors_item, colors_tag  # イベントハンドラでアクセスするためグローバル化
    
    X, U, V, colors_item, colors_tag, label1, label2 = X_input, U_input, V_input, colors_item_input, colors_tag_input, label1_input, label2_input

    latent_dim1 = U.shape[-1]
    latent_projection_type = '3d' if latent_dim1 > 2 else 'rectilinear'

    # fig1 = plt.figure(figsize=(10, 5))
    # gs = fig1.add_gridspec(1, 2)
    # latent1_ax = fig1.add_subplot(gs[0, 0], projection=latent_projection_type)
    # latent2_ax = fig1.add_subplot(gs[0, 1], projection=latent_projection_type)
    fig1 = plt.figure(figsize=(16, 8))
    gs = fig1.add_gridspec(10, 2)
    latent1_ax = fig1.add_subplot(gs[0:8, 0], projection=latent_projection_type)
    latent2_ax = fig1.add_subplot(gs[0:8, 1], projection=latent_projection_type)
    latent4_ax = fig1.add_subplot(gs[8:10,:], projection=latent_projection_type)

    fig2 = plt.figure(figsize=(5,15))
    latent3_ax = fig2.add_subplot(1,1,1)
    latent3_ax.set_axis_off()
    # latent4_ax = fig.add_subplot(gs[2,1])
    # latent4_ax.axis('off')

    #データ点のプロット
    draw_latent_2D(latent1_ax, U, label1)
    draw_latent_2D(latent2_ax, V, label2)

    # # カラーマップの設定
    # # 要素数に基づいてカラーマップを均等に割り振る
    # cmap_item = cm.get_cmap('hsv', len(X))  # カラーマップを作成
    # norm_item = Normalize(vmin=0, vmax=len(X) - 1)  # カラーマップの正規化
    # colors_item = cmap_item(norm_item(range(len(X))))  # 特に色を変えたいデータ点の色
    # cmap_tag = cm.get_cmap('hsv', len(X[-1]))  # カラーマップを作成
    # norm_tag = Normalize(vmin=0, vmax=len(X[-1]) - 1)  # カラーマップの正規化
    # colors_tag = cmap_tag(norm_tag(range(len(X[-1]))))  # 特に色を変えたいデータ点の色

    # クリックイベントを登録
    fig1.canvas.mpl_connect('button_press_event', on_click)

    plt.show()