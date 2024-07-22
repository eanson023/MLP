import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import gridspec

from mlp.utils import io_utils
from mlp.utils.plot_script import *

# no X forwarding on remote machine using ssh & screen & tmux
plt.switch_backend("agg")

try:
    import seaborn as sns

    sns.set_style("whitegrid", {"axes.grid": False})
except ImportError:
    print("Install seaborn to colorful visualization!")
except:
    print("Unknown error")

FONTSIZE = 3
plt.rc('xtick', labelsize=FONTSIZE)
plt.rc('ytick', labelsize=FONTSIZE)


def add_attention_to_figure(fig, gc, row, col, row_height, col_width, att,
                            x_labels=None, y_labels=None, title=None, aspect="auto",  # "equal"
                            show_colorbar=False, vmin=None, vmax=None, yrotation=90, cmap="binary"):
    """ helper functions for visualization """
    ax = fig.add_subplot(gc[row:row + row_height, col:col + col_width])
    iax = ax.imshow(att, interpolation="nearest", aspect=aspect,
                    cmap=cmap, vmin=vmin, vmax=vmax)
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=FONTSIZE)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, rotation=yrotation, fontsize=FONTSIZE)
    else:
        ax.get_yaxis().set_visible(False)
    if title is not None:
        ax.set_title(title)
    if show_colorbar:
        fig.colorbar(iax)
    # ax.grid()


def visualize_MLP(save_dir, vis_data, prefix, use_contrast=False):
    # fetching data
    qids = vis_data["qids"]  # [B]
    mids = vis_data["mids"]  # [B]
    query = vis_data["query"]  # [B, T, L_q] == [5,25]
    pred_s_prob = vis_data["grounding_start_prob"]  # [B, T]
    pred_e_prob = vis_data["grounding_end_prob"]  # [B, T]
    vid_nfeats = vis_data["nfeats"]
    gt_h_score = vis_data["grounding_gt"]  # [B,T]
    h_score = vis_data["highlight_score"]  # [B,T]

    # constants
    B, T = pred_s_prob.shape

    #### visualize probability and highlight score
    figsize = [10, B]  # (col, row)
    fig = plt.figure(figsize=figsize)
    gc = gridspec.GridSpec(figsize[1], figsize[0])
    gc.update(wspace=0.0, hspace=0.2)

    for i in range(B):
        ax = fig.add_subplot(gc[i:i + 1, :])
        nfeats = int(vid_nfeats[i])
        ax.plot(pred_s_prob[i, :nfeats], label='P_s', linewidth=0.5)
        ax.plot(pred_e_prob[i, :nfeats], label='P_e', linewidth=0.5)
        ax.plot(h_score[i, :nfeats], label='h_score', linewidth=0.5)
        # set highlight region
        highlight_indices = np.where(gt_h_score[i, :nfeats] == 1)[0]
        start_indices = np.diff(highlight_indices) != 1
        start_indices = np.concatenate(([True], start_indices))
        end_indices = np.diff(highlight_indices) != 1
        end_indices = np.concatenate((end_indices, [True]))
        for start, end in zip(highlight_indices[start_indices], highlight_indices[end_indices]):
            ax.fill_between(range(start, end + 1), 0, 1, color='yellow', alpha=0.2)

        ax.set_ylim(0, 1.0)
        title = f"[mid]: {mids[i]}, [qid]: {qids[i]}, [T]:{nfeats}, [query]: {query[i]}"
        ax.set_title(title, fontsize=5)

        # 调整坐标轴刻度标签的大小
        ax.tick_params(axis='both', which='both', labelsize=5)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.005, top=0.995)
    # save figure
    save_dir = os.path.join(save_dir, "qualitative", "Train")
    io_utils.check_and_create_dir(save_dir)
    save_path = os.path.join(save_dir, prefix + "_qrn.png")
    plt.savefig(save_path, dpi=300)
    print("Visualization of Teslam is saved in {}".format(save_path))
    plt.close()

    if not use_contrast:
        return

    #### visualize contrastive similarity
    tmr = vis_data["contrastive_tmr"]
    mtr = vis_data["contrastive_mtr"]

    figsize = [9, 3 * 2]  # (col, row)
    fig = plt.figure(figsize=figsize)
    gc = gridspec.GridSpec(figsize[1], figsize[0])
    gc.update(wspace=0.5, hspace=1.0)

    ax = fig.add_subplot(gc[0:2, 0:4])
    ax.axis('off')
    texts = [f"{i:0>2} [mid]: {mids[i]:<5}, [qid]: {qids[i]:<5}, [text]: {query[i]}" for i in range(B // 2)]
    ax.text(0.0, 0.5, "\n".join(texts), fontsize=8, ha='left', va='center')

    ax = fig.add_subplot(gc[0:2, 5:9])
    ax.axis('off')
    texts = [f"{i:0>2} [mid]: {mids[i]:<5}, [qid]: {qids[i]:<5}, [text]: {query[i]}" for i in range(B // 2, B)]
    ax.text(0.0, 0.5, "\n".join(texts), fontsize=8, ha='left', va='center')

    labels = np.arange(len(tmr[0]))
    add_attention_to_figure(fig, gc, 2, 0, 2, 3, tmr[0], title="text-to-motion: cosine_similarity",
                            show_colorbar=True, x_labels=labels, y_labels=labels, vmin=-1, vmax=1, aspect="equal",
                            cmap="coolwarm")
    add_attention_to_figure(fig, gc, 2, 3, 2, 3, tmr[1], title="weight_matrix (anti text-modal similarity)",
                            show_colorbar=True, x_labels=labels, y_labels=labels, aspect="equal", cmap="viridis")
    add_attention_to_figure(fig, gc, 2, 6, 2, 3, tmr[2], x_labels=labels, y_labels=labels,
                            title="weighted_logits",
                            show_colorbar=True, aspect="equal", cmap="coolwarm")

    add_attention_to_figure(fig, gc, 4, 0, 2, 3, mtr[0], x_labels=labels, y_labels=labels,
                            title="motion-to-text: cosine_similarity",
                            show_colorbar=True, vmin=-1, vmax=1, aspect="equal", cmap="coolwarm")
    add_attention_to_figure(fig, gc, 4, 3, 2, 3, mtr[1], x_labels=labels, y_labels=labels,
                            title="weight_matrix (anti motion-modal similarity)",
                            show_colorbar=True, aspect="equal", cmap="viridis")
    add_attention_to_figure(fig, gc, 4, 6, 2, 3, mtr[2], x_labels=labels, y_labels=labels,
                            title="weighted_logits",
                            show_colorbar=True, aspect="equal", cmap="coolwarm")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    save_path = os.path.join(save_dir, prefix + "_con.png")
    plt.savefig(save_path, dpi=300)
    print("Visualization of Teslam is saved in {}".format(save_path))
    plt.close()

def visualize_MLP_compare(save_dir, vis_data, prefix, use_contrast=False):
    # fetching data
    qids = vis_data["qids"]  # [B]
    mids = vis_data["mids"]  # [B]
    query = vis_data["query"]  # [B, T, L_q] == [5,25]
    pred_s_prob_base = vis_data["grounding_start_prob"]  # [B, T]
    pred_e_prob_base = vis_data["grounding_end_prob"]  # [B, T]
    vid_nfeats = vis_data["nfeats"]
    gt_h_score = vis_data["grounding_gt"]  # [B,T]
    h_score_base = vis_data["highlight_score"]  # [B,T]

    pred_s_prob_new = vis_data["grounding_start_prob_new"]  # [B, T]
    pred_e_prob_new = vis_data["grounding_end_prob_new"]  # [B, T]
    h_score_new = vis_data["highlight_score_new"]  # [B,T]
    pred_s_prob_new_dn = vis_data["grounding_start_prob_dn"]
    pred_e_prob_new_dn = vis_data["grounding_end_prob_dn"]

    # constants
    B, T = pred_s_prob_base.shape

    #### visualize probability and highlight score
    figsize = [10, B]  # (col, row)
    fig = plt.figure(figsize=figsize)
    gc = gridspec.GridSpec(figsize[1], figsize[0])
    gc.update(wspace=0.0, hspace=0.2)

    for i in range(B):
        ax = fig.add_subplot(gc[i:i + 1, :])
        nfeats = int(vid_nfeats[i])
        ax.plot(pred_s_prob_base[i, :nfeats], label='P_s', linewidth=0.1)
        ax.plot(pred_e_prob_base[i, :nfeats], label='P_e', linewidth=0.1)
        # ax.plot(h_score_base[i, :nfeats], label='h_score', linewidth=0.1)


        ax.plot(pred_s_prob_new[i, :nfeats], label='P_s_new', linewidth=0.5)
        ax.plot(pred_e_prob_new[i, :nfeats], label='P_e_new', linewidth=0.5)
        # ax.plot(h_score_new[i, :nfeats], label='h_score_new', linewidth=0.5)

        ax.plot(pred_s_prob_new_dn[i, :nfeats], label='P_s_new', linewidth=1)
        ax.plot(pred_e_prob_new_dn[i, :nfeats], label='P_e_new', linewidth=1)
        # set highlight region
        highlight_indices = np.where(gt_h_score[i, :nfeats] == 1)[0]
        start_indices = np.diff(highlight_indices) != 1
        start_indices = np.concatenate(([True], start_indices))
        end_indices = np.diff(highlight_indices) != 1
        end_indices = np.concatenate((end_indices, [True]))
        max_high = max(np.max(pred_s_prob_base[i, :nfeats]), np.max(pred_e_prob_base[i, :nfeats]))
        max_high = max(max_high,np.max(pred_s_prob_new[i, :nfeats]))
        max_high = max(max_high, np.max(pred_s_prob_new_dn[i, :nfeats]))
        max_high = max(max_high, np.max(pred_e_prob_new_dn[i, :nfeats]))
        for start, end in zip(highlight_indices[start_indices], highlight_indices[end_indices]):
            ax.fill_between(range(start, end + 1), 0, max_high, color='yellow', alpha=0.2)

        # ax.set_ylim(0, 1.0)
        title = f"[mid]: {mids[i]}, [qid]: {qids[i]}, [T]:{nfeats}, [query]: {query[i]}"
        ax.set_title(title, fontsize=5)

        # 调整坐标轴刻度标签的大小
        ax.tick_params(axis='both', which='both', labelsize=5)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.005, top=0.995)
    # save figure
    save_dir = os.path.join(save_dir, "qualitative", "Train")
    io_utils.check_and_create_dir(save_dir)
    save_path = os.path.join(save_dir, prefix + "_qrn.png")
    plt.savefig(save_path, dpi=300)
    print("Visualization of Teslam is saved in {}".format(save_path))
    plt.close()