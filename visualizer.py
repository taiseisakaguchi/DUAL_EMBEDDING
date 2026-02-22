import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.animation import FuncAnimation
import japanize_matplotlib
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CONST = 10

def visualize_history(U_history, V_history, error_history, visual_tag_list, visual_object_list, sub_tag_vec, sub_object_vec, colors_item, colors_tag, label1='None', label2='None', save_mp4=True, filename="tmp"):
    latent_dim1, latent_dim2 = U_history.shape[-1], V_history.shape[-1]
    latent_projection_type = '3d' if latent_dim1 > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    latent1_ax = fig.add_subplot(gs[0:2, 0], projection=latent_projection_type)
    latent2_ax = fig.add_subplot(gs[0:2, 1], projection=latent_projection_type)
    error_ax = fig.add_subplot(gs[2, :])
    num_epoch = len(error_history)

    latent1_drawer = [None, None, draw_latent_2D, draw_latent_3D][latent_dim1]
    latent2_drawer = [None, None, draw_latent_2D, draw_latent_3D][latent_dim2]
    ani = FuncAnimation(
        fig,
        update_graph,
        frames=num_epoch//CONST,
        repeat=True,
        interval=100,
        fargs=(latent1_drawer, latent2_drawer, U_history, V_history, error_history, visual_tag_list, visual_object_list, sub_tag_vec, sub_object_vec, colors_item, colors_tag, fig, latent1_ax, latent2_ax, error_ax, label1, label2)
    )

    plt.show()

    # if save_mp4:
        # ani.save(f"{filename}.gif", writer="pillow")
        # ani.save(f"{filename}.mp4", writer="ffmpeg")
        #DropBox上でプログラムを動かすならこれ↓
        # ani.save(f"/Users/furukawashuushi/Desktop/{filename}.mp4",writer="ffmpeg")

def update_graph(epoch, latent1_drawer, latent2_drawer, U_history, V_history, error_history, visual_tag_list, visual_object_list, sub_tag_vec, sub_object_vec, colors_item, colors_tag, fig, latent1_ax, latent2_ax, error_ax, label1, label2):
    #この関数内だったらforループを回しても問題ないが、この関数内で呼び出した関数の中でforループを使うと問題有り
    fig.suptitle(f"epoch: {epoch*CONST}")
    latent1_ax.cla()
    latent2_ax.cla()
    error_ax.cla()

    U, V = U_history[epoch*CONST], V_history[epoch*CONST]

    latent1_drawer(latent1_ax, U, label1, colors_item, visual_object_list)
    latent2_drawer(latent2_ax, V, label2, colors_tag, visual_tag_list)

    if visual_tag_list is not None:
        #部分空間のプロット
        # 配列の次元に応じて処理を変える
        if len(sub_tag_vec.shape) == 1:  # 1×mの場合
            Visualizer_subspace(latent1_ax, sub_tag_vec)  # 直接渡す
        else:  # n×mの場合
            for i, vector in enumerate(sub_tag_vec):
                Visualizer_subspace(latent1_ax, vector, colors_tag[visual_tag_list[i]])
    if visual_object_list is not None:
        if len(sub_object_vec.shape) == 1:  # 1×mの場合
            Visualizer_subspace(latent2_ax, sub_object_vec)  # 直接渡す
        else:  # n×mの場合
            for i, vector in enumerate(sub_object_vec):
                Visualizer_subspace(latent2_ax, vector, colors_item[visual_object_list[i]])
    draw_error(error_ax, error_history, epoch*CONST)

def draw_latent_2D(ax, Z, label, color, list):
    if list is not None:
        # 1. listに含まれないデータ点をフィルタリングして描画
        mask = np.ones(len(Z), dtype=bool)  # 全データ点を True に初期化
        mask[list] = False  # list に含まれるデータ点を False に設定
        ax.scatter(Z[mask, 0], Z[mask, 1], color='k', label=None)  # 黒で描画
        # 2. list に含まれるデータ点を個別に描画
        for i, idx in enumerate(list):
            ax.scatter(Z[idx, 0], Z[idx, 1], color=color[idx], label=label[idx])
        # for i, tag in enumerate(list):
        #     ax.scatter(Z[tag, 0], Z[tag, 1], color = color[i], label =label[tag])
        # for i in list: #指定した部分空間のベクトルの名前を可視化したいならこっち
        #     ax.text(Z[i, 0], Z[i, 1], label[i], path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
        # ax.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=12)
        for i, name in enumerate(label): #全ベクトルの名前を可視化したいならこっち
             ax.text(Z[i, 0], Z[i, 1], name, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
        # for i in range(len(label)): #全ベクトルの番号を可視化
        #     ax.text(Z[i, 0], Z[i, 1], i, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
    elif list is None:
        ax.scatter(Z[:, 0], Z[:, 1], color = 'k')
        # for i in range(len(label)): #全ベクトルの番号を可視化
        #     ax.text(Z[i, 0], Z[i, 1], i, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
        for i, name in enumerate(label): #全ベクトルの名前を可視化したいならこっち
             ax.text(Z[i, 0], Z[i, 1], name, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    ax.set_xlim(-(lim + 0.05), lim + 0.05)
    ax.set_ylim(-(lim + 0.05), lim + 0.05)
    ax.set(aspect='equal')

def draw_latent_3D(ax, Z, label, color, list):
    # ax.set_zlim3d(-2, 2)
    if list is not None:
        # 1. listに含まれないデータ点をフィルタリングして描画
        mask = np.ones(len(Z), dtype=bool)  # 全データ点を True に初期化
        mask[list] = False  # list に含まれるデータ点を False に設定
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], color='k', label=None)  # 黒で描画
        # ax.quiver(0, 0, 0, Z[mask, 0], Z[mask, 1], Z[mask, 2], color='gray', linestyle='--', arrow_length_ratio=0.0, alpha=0.3)
        # 2. list に含まれるデータ点を個別に描画
        for i, idx in enumerate(list):
            ax.scatter(Z[idx, 0], Z[idx, 1], Z[idx, 2], color=color[idx], label=label[idx])
            # ax.quiver(0, 0, 0, Z[idx, 0], Z[idx, 1], Z[idx, 2], color=color[idx], linestyle='--', arrow_length_ratio=0.0, alpha=0.7)
        for i, name in enumerate(label): #全ベクトルの名前を可視化したいならこっち
            ax.text(Z[i, 0], Z[i, 1],  Z[i, 2], name, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
        # for i in list: #指定した部分空間のベクトルの名前を可視化したいならこっち
        #     ax.text(Z[i, 0], Z[i, 1], Z[i, 2], label[i], path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
        # ax.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=12)
    elif list is None:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], color ='k')
        # ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], color =color)
        # for i in range(len(label)): #全ベクトルの番号を可視化
        #     ax.text(Z[i, 0], Z[i, 1], Z[i, 2], i, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
        for i, name in enumerate(label): #全ベクトルの名前を可視化したいならこっち
             ax.text(Z[i, 0], Z[i, 1], Z[i, 2],name, path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")])
            # ax.quiver(0, 0, 0, Z[i, 0], Z[i, 1], Z[i, 2], color='gray', linestyle='--', arrow_length_ratio=0.0, alpha=0.3)


def Visualizer_subspace(ax, sub_vec, color):
    if sub_vec.shape[-1] == 2:
        plot_line(ax, sub_vec, color)
    elif sub_vec.shape[-1] == 3:
        # plane1 = calculate_plane(sub_vec)
        plot_planes(ax, sub_vec, color)

def plot_line(ax, sub_vec, color):
    # ax.quiver(0, 0, sub_vec[0], sub_vec[1], angles='xy', scale_units='xy', scale=1, color=color)
    sub_vec = sub_vec/np.linalg.norm(sub_vec)**2
    # 直線を描画
    ax.axline((0, (sub_vec[0]**2 + sub_vec[1]**2)/sub_vec[1]), slope = -(sub_vec[0] / sub_vec[1]), color =color)
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)

def plot_planes(ax, sub_vec, color, resolution=50):
    normalized_sub_vec = sub_vec/np.linalg.norm(sub_vec)
    nx, ny, nz= normalized_sub_vec
    a, b, c = sub_vec
    d=1

    x = np.linspace(-2,2, 2)
    y = np.linspace(-2,2, 2)
    X, Y = np.meshgrid(x, y)
    Z = (d - a * X - b * Y) / c
    # 平面1をプロット
    ax.plot_surface(X, Y, Z, color = color, alpha=0.3)#alphaは透明度

    # # 楕円状のメッシュを作成（XY平面上で楕円を作成）
    # theta = np.linspace(0, 2*np.pi, resolution)
    # X_circle = 1.5* np.cos(theta)
    # Y_circle = 1 * np.sin(theta)
    # # 楕円を法線ベクトルに対して回転
    # # 法線 v = (a, b, c) に垂直なベクトル2つを求める
    # u1 = np.array([-b, a, 0])  # 法線と直交するベクトル1
    # u2 = np.cross(normalized_sub_vec, u1)  # 法線と u1 に垂直なベクトル2
    # u1 /= np.linalg.norm(u1)
    # u2 /= np.linalg.norm(u2)
    # # 楕円を平面上に投影
    # X_plane = u1[0] * X_circle + u2[0] * Y_circle + d * normalized_sub_vec[0]
    # Y_plane = u1[1] * X_circle + u2[1] * Y_circle + d * normalized_sub_vec[1]
    # Z_plane = u1[2] * X_circle + u2[2] * Y_circle + d * normalized_sub_vec[2]
    # # 平面（楕円）を描画
    # verts = [list(zip(X_plane, Y_plane, Z_plane))]
    # ax.add_collection3d(Poly3DCollection(verts, color=color, alpha=0.3))
    # ベクトル v をプロット
    ax.quiver(0, 0, 0, nx, ny, nz, color=color, arrow_length_ratio=0.0)
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5
    # mid_x = (X.max()+X.min()) * 0.5
    # mid_y = (Y.max()+Y.min()) * 0.5
    # mid_z = (Z.max()+Z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(-1.5, 1.5)

def draw_error(ax, error_history, epoch):
    ax.set_title("error_function", fontsize=8)
    ax.plot(error_history, label='誤差関数')
    ax.scatter(epoch, error_history[epoch], s=55, marker="*")
    ax.text
    ax.set_ylim(bottom=0)

def pause_plot(U_history, V_history, error_history, visual_tag_list, visual_object_list, sub_tag_vec, sub_object_vec, colors_item, colors_tag, label1='None', label2='None'):
    latent_dim1 = U_history.shape[-1]
    latent_projection_type = '3d' if latent_dim1 > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    latent1_ax = fig.add_subplot(gs[0:2, 0], projection=latent_projection_type)
    latent2_ax = fig.add_subplot(gs[0:2, 1], projection=latent_projection_type)
    error_ax = fig.add_subplot(gs[2, :])

    #データ点のプロット
    if latent_projection_type == 'rectilinear':
        draw_latent_2D(latent1_ax, U_history, label1, colors_item, visual_object_list)
        draw_latent_2D(latent2_ax, V_history, label2, colors_tag, visual_tag_list)
    elif latent_projection_type == '3d' :
        draw_latent_3D(latent1_ax, U_history, label1, colors_item, visual_object_list)
        draw_latent_3D(latent2_ax, V_history, label2, colors_tag, visual_tag_list)

    if visual_tag_list is not None:
        #部分空間のプロット
        # 配列の次元に応じて処理を変える
        if len(sub_tag_vec.shape) == 1:  # 1×mの場合
            Visualizer_subspace(latent1_ax, sub_tag_vec)  # 直接渡す
        else:  # n×mの場合
            for i, vector in enumerate(sub_tag_vec):
                Visualizer_subspace(latent1_ax, vector, colors_tag[visual_tag_list[i]])
    if visual_object_list is not None:
        if len(sub_object_vec.shape) == 1:  # 1×mの場合
            Visualizer_subspace(latent2_ax, sub_object_vec)  # 直接渡す
        else:  # n×mの場合
            for i, vector in enumerate(sub_object_vec):
                Visualizer_subspace(latent2_ax, vector, colors_item[visual_object_list[i]])
    error_ax.plot(error_history)
    error_ax.set_ylim(bottom=0)
    plt.pause(0.3)
    plt.close()

import seaborn as sns
import pandas as pd
def ShowCosinSimirarity(X, texts):
    norm_vector = np.linalg.norm(X, axis=1)  # 行ごとのノルム
    X_normed = X / norm_vector[:, None]
    cosine_similarity_matrix = X_normed @ X_normed.T
    df = pd.DataFrame(data=cosine_similarity_matrix, index = texts, columns = texts, dtype=float)
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        df,
        annot=True,        # 各セルに数値を表示したい場合は True
        fmt=".2f",         # 小数点以下2桁表示
        cmap="bwr",
        mask=np.triu(np.ones_like(df, dtype=bool)),
        xticklabels = 1,
        yticklabels = 1
    )
    plt.title("Cosine Similarity Matrix")

def VisualResult(U, V, error, visual_tag_list, visual_object_list, sub_tag_vec, sub_object_vec, colors_item, colors_tag, label1='None', label2='None'):
    latent_dim1 = U.shape[-1]
    latent_projection_type = '3d' if latent_dim1 > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    latent1_ax = fig.add_subplot(gs[0:2, 0], projection=latent_projection_type)
    latent2_ax = fig.add_subplot(gs[0:2, 1], projection=latent_projection_type)
    error_ax = fig.add_subplot(gs[2, :])

    #データ点のプロット
    if latent_projection_type == 'rectilinear':
        draw_latent_2D(latent1_ax, U, label1, colors_item, visual_object_list)
        draw_latent_2D(latent2_ax, V, label2, colors_tag, visual_tag_list)
    elif latent_projection_type == '3d' :
        draw_latent_3D(latent1_ax, U, label1, colors_item, visual_object_list)
        draw_latent_3D(latent2_ax, V, label2, colors_tag, visual_tag_list)

    if visual_tag_list is not None:
        #部分空間のプロット
        # 配列の次元に応じて処理を変える
        if len(sub_tag_vec.shape) == 1:  # 1×mの場合
            Visualizer_subspace(latent1_ax, sub_tag_vec)  # 直接渡す
        else:  # n×mの場合
            for i, vector in enumerate(sub_tag_vec):
                Visualizer_subspace(latent1_ax, vector, colors_tag[visual_tag_list[i]])
    if visual_object_list is not None:
        if len(sub_object_vec.shape) == 1:  # 1×mの場合
            Visualizer_subspace(latent2_ax, sub_object_vec)  # 直接渡す
        else:  # n×mの場合
            for i, vector in enumerate(sub_object_vec):
                Visualizer_subspace(latent2_ax, vector, colors_item[visual_object_list[i]])
    error_ax.plot(error)
    error_ax.set_ylim(bottom=0)
    ShowCosinSimirarity(V, label2)
    plt.tight_layout()
    plt.show()