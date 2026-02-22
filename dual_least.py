import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import datetime
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.cm as cm

# from DataCreate import load_data
from visualizer import VisualResult
from ReadData import load_data
from ClickView_1D import Click_and_View_1D
from ProjectionCS import ShowComplementarySubspace

now = datetime.datetime.now()
# np.random.seed(42)

beta = 1.0
def func0116(X, U, V, C = 0.2, a=1/10):#坂口の修論はこれ
    d = ((U @ V.T) - 1)**2
    cos_sim =(U @ V.T)/(jnp.linalg.norm(U)*jnp.linalg.norm(V))
    return beta/2 *jnp.sum(X * d) - jnp.sum((1-X) * jnp.log(1 - C * jnp.exp((-beta/2) * d))) - a * jnp.sum(X*cos_sim**2)

def E(X, U, V, epoch):
    return func0116(X, U, V)

def fit(epoch, eta_u, eta_v, X, U, V, visual_tag_list, visual_object_list, colors_item, colors_tag):
    history = {}
    history['U'] = np.zeros((epoch, U.shape[0], U.shape[1]))
    history['V'] = np.zeros((epoch, V.shape[0], V.shape[1]))
    history['E'] = np.zeros((epoch, 1))

    for e in tqdm(range(epoch)):
        grad_u = jax.grad(E, argnums=1)(X, U, V, e)
        grad_v = jax.grad(E, argnums=2)(X, U, V, e)

        U = U - eta_u * grad_u
        V = V - eta_v * grad_v

        history['U'][e] = U
        history['V'][e] = V
        history['E'][e] = E(X, U, V, e)

        #学習途中で損失関数を確認したいときに一時的に表示
        # if(e%100 == 0):
            # pause_plot_MBFCH(e, history['U'][e], history['V'][e], history['E'], special_item, special_tag, history['V'][e, visual_tag_list, :], history['U'][e, visual_object_list, :], label1=object_labels, label2=tag_labels)
            # pause_plot(history['U'][e], history['V'][e], history['E'],visual_tag_list, visual_object_list, history['V'][e, visual_tag_list, :], history['U'][e, visual_object_list, :], colors_item, colors_tag, label1=object_labels, label2=tag_labels)
    return history

if __name__ == '__main__':
    data = load_data(retlabel_object=True, retlabel_tag=True)
    X = data[0]
    object_labels = data[1]
    tag_labels = data[2]

    #動物3種のデータ使うなら800,0.01,0.01の設定がいい感じ。
    epoch = 800
    eta_u = 0.01/beta
    eta_v = 0.01/beta
    latent_dim_u = 3
    latent_dim_v = 3

    # 可視化したい部分空間をここで指定する
    # visual_object_list = None #オブジェクトの部分空間を何も可視化しないならこれ
    # visual_tag_list = None #タグの部分空間を何も可視化しないならこれ
    visual_object_list = np.arange(len(X)) #オブジェクトの部分空間を全部可視化したいとき
    visual_tag_list = np.arange(len(X[-1])) #タグの部分空間を全部可視化したいとき

    # U = np.random.uniform(-1, 1, X.shape[0]*latent_dim_u).reshape(X.shape[0], latent_dim_u)
    # V = np.random.uniform(-1, 1, X.shape[1]*latent_dim_v).reshape(X.shape[1], latent_dim_v)
    U = np.random.normal(loc = 0, scale = 0.005, size = X.shape[0]*latent_dim_u).reshape(X.shape[0], latent_dim_u)
    V = np.random.normal(loc = 0, scale = 0.005, size = X.shape[1]*latent_dim_v).reshape(X.shape[1], latent_dim_v)

    # カラーマップの設定
    cmap_item = cm.get_cmap('hsv', len(X))  # カラーマップを作成
    norm_item = Normalize(vmin=0, vmax=len(X) - 1)  # カラーマップの正規化
    colors_item = cmap_item(norm_item(range(len(X))))  # 特に色を変えたいデータ点の色
    # cmap_tag = cm.get_cmap('hsv', len(X[-1]))  # カラーマップを作成
    # norm_tag = Normalize(vmin=0, vmax=len(X[-1]) - 1)  # カラーマップの正規化
    # colors_tag = cmap_tag(norm_tag(range(len(X[-1]))))  # 特に色を変えたいデータ点の色

    visual_object_list = [13,14,0,5] #タグ空間で表示したいオブジェクト番号を指定
    # visual_object_list = [13,14] #タグ空間で表示したいオブジェクト番号を指定
    #0:ハト=b、5:タカ=o、13:ライオン=r、14:ウマ=g
    visual_tag_list = [17,16,5,4] #オブジェクト空間で表示したいタグ番号を指定
    # visual_tag_list = [17,16] #オブジェクト空間で表示したいタグ番号を指定
    #4:二本足=o、5:四本脚=b、16:草食=g、17:肉食=r
    group_colors = ['red', 'green', 'blue','orange']
    # group_colors_i = ['red','green']
    # group_colors_t = ['red','green']

    colors_item = np.array(['k'] * len(X),dtype=object)
    colors_tag = np.array(['k'] * len(X[-1]),dtype=object)
    colors_item[visual_object_list], colors_tag[visual_tag_list]=group_colors, group_colors

    history = fit(epoch, eta_u, eta_v, X, U, V, visual_tag_list, visual_object_list, colors_item, colors_tag)
    history.update({'X': X.tolist(), 'object': object_labels.tolist(), 'tag': tag_labels.tolist()})
    # np.save(f"history_{now.strftime('%Y%m%d_%H%M%S')}.npy", history)

    sub_tag_vec = history['V'][-1, visual_tag_list, :] #タグ部分空間の可視化のためのベクトル
    sub_object_vec = history['U'][-1, visual_object_list, :] #オブジェクト部分空間の可視化のためのベクトル

    VisualResult(history['U'][-1, :, :], history['V'][-1, :, :], history['E'], visual_tag_list, visual_object_list, sub_tag_vec, sub_object_vec, colors_item, colors_tag, label1=object_labels, label2=tag_labels)

    if latent_dim_u == 2 and latent_dim_v == 2:
        Click_and_View_1D(X, history['U'][-1, :, :], history['V'][-1, :, :], colors_item, colors_tag, label1_input=object_labels, label2_input = tag_labels)
    if latent_dim_u == 3 and latent_dim_v == 3:
        ShowComplementarySubspace(history['U'][-1, :, :], history['V'][-1, :,:], colors_tag, X, object_labels, tag_labels) #1つの視点で垂直射影するならこれ