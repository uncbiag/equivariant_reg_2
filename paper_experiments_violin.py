import torch
import numpy as np
import footsteps
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
line_styles = ["solid", "dashed", "dotted"]

name_map = {"baseline":"train baseline\ntest baseline",
        "tolerate shift":"train shift\ntest shift",
        "generalize shift":"train baseline\n test shift",
        "hard mode":"train scale shift\ntest scale shift",
        "hard mode generalize":"train baseline\n test scale shift"}


def adjust_title(title):
    if not "trans" in title:
        return ""
    title = title.replace("_network", "")
    title = title.replace("make_", "")
    title = title.replace("just_", "")
    title = title.split(" ")[0]
    title = title.replace("_", " ")

    return name_map[title]

experiments = torch.load("results/dice_data_paper_expr-4/dice_results.trch")


violin_plot = plt.violinplot(experiments.values(), bw_method="silverman")

plt.title("Distribution of Test DICE")

hatches = [None, "-", "."]

for i, pc in enumerate(violin_plot["bodies"]):

    pc.set_facecolor(colors[i % 3])
    
    #if hatches[i % 3]:
    #    pc.set_hatch(hatches[i % 3])
    #    pc.set_edgecolor(colors[i // 3])
    

#violin_plot["cbars"].set_linestyles(line_styles * 3)

violin_plot["cbars"].set_colors([colors[i % 3] for i in range(15)])
violin_plot["cmins"].set_colors([colors[i % 3] for i in range(15)])
violin_plot["cmaxes"].set_colors([colors[i % 3] for i in range(15)])

#plt.gca().tick_params(axis='x', colors='white')
#plt.xticks(rotation=90)
def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
set_axis_style(plt.gca(), list(map(adjust_title, experiments.keys())))
plt.legend(["Hybrid", "Transformer", "U-Net"], ncols=3, loc="lower left")
plt.ylim([0, .8])
plt.tight_layout()

footsteps.plot("violin.png")
