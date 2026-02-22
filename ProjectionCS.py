import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.patheffects as patheffects
from adjustText import adjust_text
import tkinter as tk
from tkinter import ttk, messagebox

def Projection1D(X, ax_data, view_vec, color, label, title):
    fig = plt.figure(figsize=(16, 2))
    ax = fig.add_subplot(111)
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
    ax.set_title(f"{title}の視点でみたアイテム類似度(1次元射影)")
    plt.show(block=False)  # ブロックしない

def RectangularCoordinateSystem(X, U, V, selection1, selection2, label, title):
    fig = plt.figure(figsize=(10, 10))
    vec1 = V[selection1, :]
    vec2 = V[selection2, :]
    ax = fig.add_subplot(111)
    d1 = np.sum(U * vec1, axis=1)/np.linalg.norm(vec1)
    d2 = np.sum(U * vec2, axis=1)/np.linalg.norm(vec2)
    
    frag_r, frag_b, frag_star = True, True, True
    for i, x in enumerate(X):
        if x[selection1] != 0 and x[selection2] == 0:
            ax.scatter(d1[i], d2[i], color='r', label=title[selection1] if frag_r else "")
            frag_r = False
        elif x[selection1] == 0 and x[selection2] != 0:
            ax.scatter(d1[i], d2[i], color='b', label=title[selection2] if frag_b else "")
            frag_b = False
        elif x[selection1] != 0 and x[selection2] != 0:
            ax.scatter(d1[i], d2[i], color='g', marker='*', 
                      label=f'{title[selection1]}と{title[selection2]}' if frag_star else "")
            frag_star = False
        else:
            ax.scatter(d1[i], d2[i], color='k')
    
    texts = [ax.text(d1[i], d2[i], label[i], fontsize=12,
                     path_effects=[patheffects.withStroke(linewidth=2, foreground='white', capstyle="round")]) 
             for i in range(len(X))]
    adjust_text(texts)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'{title[selection1]}の視点')
    ax.set_ylabel(f'{title[selection2]}の視点')
    ax.legend(loc='upper right', fontsize=10)
    plt.show(block=False)
    plt.pause(0.001)

# グラム・シュミットの正規直交化法
def GramSchmidtorthonormalization(x, n):
    # ----- 1) まずデータと前処理 -----
    # n を正規化 (単位ベクトル化)
    n =n.reshape(1, -1)
    norm_n = np.linalg.norm(n)
    n_hat = n / norm_n  # n_hat は単位ベクトル
    n_tile =np.tile(n_hat,(len(x),1))
    # x を n に射影したベクトル x' = (x^T n_hat) n_hat
    x_prime = np.dot(x, n_hat.T) * n_tile

    # 平面上に潰したベクトル y = x - x' （= n方向成分を除いたもの）
    y = x - x_prime

    # ----- 2) n と直交する 2次元空間の基底ベクトル e1, e2 を作る -----
    #  - n_hat と別ベクトル w の外積で e1 を作り、正規化する
    #  - その後 e2 = n_hat x e1 とする
    #  - w は n_hat と平行でないベクトルなら何でもよい
    w = np.array([0.0, 0.0, 1.0])
    # もし n_hat と w がほぼ平行なら、別の w に切り替え
    if np.allclose(n_hat, w / np.linalg.norm(w)):
        w = np.array([0.0, 1.0, 0.0])

    e1_unnorm = np.cross(n_hat, w)
    e1_unnorm_norm = np.linalg.norm(e1_unnorm)
    e1 = e1_unnorm / e1_unnorm_norm  # 正規化

    # e2 = n_hat x e1
    e2 = np.cross(n_hat, e1)

    # ----- 3) 2次元座標 (u, v) を求める -----
    #      y = x - x' の e1, e2 成分
    u = np.dot(y, e1.T)
    v = np.dot(y, e2.T)
    return u, v

def ProjectionComplementarySubspace(u, v, color, list, label, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(u, v, color = 'k')
    for i ,x in enumerate(list):
        if x !=0:
            ax.scatter(u[i], v[i], color = color)
    texts = [ax.text(u[i], v[i], label[i], fontsize=12) for i in range(len(u))]
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('e1')
    ax.set_ylabel('e2')
    ax.set_title(f"{title}の視点でみた補空間")
    adjust_text(texts)
    plt.show(block=False)  # ブロックしない


def ShowComplementarySubspace(data, V, color, X, label, title):    
    def on_select(event):
        """プルダウン選択時に呼ばれるコールバック関数"""
        selected_index = combo.current()
        if selected_index >= 0:
            u, v = GramSchmidtorthonormalization(data, V[selected_index])
            ProjectionComplementarySubspace(u, v, color[selected_index], 
                                           X[:, selected_index], label, 
                                           title[selected_index])
            Projection1D(X[:, selected_index], data, V[selected_index], 
                        color[selected_index], label, title[selected_index])
            plt.pause(0.001)  # matplotlibウィンドウを更新するための短い待機
    
    def on_close():
        """終了ボタンが押されたときの処理"""
        plt.close('all')  # すべてのmatplotlibウィンドウを閉じる
        root.destroy()
    
    # tkinterウィンドウの作成
    root = tk.Tk()
    root.title("View Point 選択")
    root.geometry("400x150")
    
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(expand=True, fill=tk.BOTH)
    
    label_widget = tk.Label(frame, text="可視化するView Pointを選択してください:", 
                           font=("Arial", 11))
    label_widget.pack(pady=(0, 10))
    
    combo = ttk.Combobox(frame, values=title, state="readonly", 
                         font=("Arial", 10), width=35)
    combo.pack(pady=(0, 15))
    
    combo.bind("<<ComboboxSelected>>", on_select)
    
    close_button = tk.Button(frame, text="終了", command=on_close, 
                            font=("Arial", 10), width=10)
    close_button.pack()
    
    root.mainloop()

def ShowRectangularCoordinateSystem(X, U, V, label, title):
    # tkinterのGUIで2つのview pointを選択し、直交座標系で可視化する関数

    def on_execute():
        """実行ボタンが押されたときの処理"""
        index1 = combo1.current()
        index2 = combo2.current()
        
        # 選択がされているかチェック
        if index1 < 0 or index2 < 0:
            messagebox.showwarning("警告", "両方のView Pointを選択してください")
            return
        
        # 可視化処理を実行
        RectangularCoordinateSystem(X, U, V, index1, index2, label, title)
    
    def on_close():
        """終了ボタンが押されたときの処理"""
        plt.close('all')  # すべてのmatplotlibウィンドウを閉じる
        root.destroy()
    
    # tkinterウィンドウの作成
    root = tk.Tk()
    root.title("View Point 選択")
    root.geometry("450x200")
    
    # フレームの作成
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(expand=True, fill=tk.BOTH)
    
    # ラベルとプルダウン1の作成
    label1 = tk.Label(frame, text="1つ目のView Point (X軸):", font=("Arial", 11))
    label1.pack(pady=(0, 5))
    
    combo1 = ttk.Combobox(frame, values=title, state="readonly", 
                          font=("Arial", 10), width=35)
    combo1.pack(pady=(0, 15))
    
    # ラベルとプルダウン2の作成
    label2 = tk.Label(frame, text="2つ目のView Point (Y軸):", font=("Arial", 11))
    label2.pack(pady=(0, 5))
    
    combo2 = ttk.Combobox(frame, values=title, state="readonly", 
                          font=("Arial", 10), width=35)
    combo2.pack(pady=(0, 15))
    
    # ボタンフレームの作成
    button_frame = tk.Frame(frame)
    button_frame.pack()
    
    # 実行ボタンの作成
    execute_button = tk.Button(button_frame, text="実行", command=on_execute, 
                              font=("Arial", 10), width=10, bg="#4CAF50", fg="white")
    execute_button.pack(side=tk.LEFT, padx=5)
    
    # 終了ボタンの作成
    close_button = tk.Button(button_frame, text="終了", command=on_close, 
                            font=("Arial", 10), width=10)
    close_button.pack(side=tk.LEFT, padx=5)
    
    # メインループの開始
    root.mainloop()