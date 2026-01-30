import matplotlib.pyplot as plt
import numpy as np

## Coded BER
#info bit 3
# data = {
#     1: {
#         "SC-NCJT": {"ber": 0.0, "mod": 6, "rate": 0.5},
#         "CJT (Ideal scenario)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#     }
# }

# #info bit 4
# data = {
#     1: {
#         "SC-NCJT": {"ber": 0.0, "mod": 8, "rate": 0.5},
#         "MC-NCJT": {"ber": 3.3908420138888885e-07, "mod": 4, "rate": 0.5},
#         "CJT (Ideal scenario)": {"ber": 0.0, "mod": 4, "rate": 0.33},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 4, "rate": 0.33},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 4, "rate": 0.33},
#     },
#     2: {
#         "SC-NCJT": {"ber": 0.0, "mod": 8, "rate": 0.5},
#         "MC-NCJT": {"ber": 0.0, "mod": 4, "rate": 0.5},
#         "CJT (Ideal scenario)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#     },
#     4: {
#         "SC-NCJT": {"ber": 0.0, "mod": 8, "rate": 0.5},
#         "MC-NCJT": {"ber": 1.6954210069444442e-07, "mod": 4, "rate": 0.5},
#         "CJT (Ideal scenario)": {"ber": 0.0, "mod": 2, "rate": 0.33},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 2, "rate": 0.33},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 2, "rate": 0.33},
#     }
# }


# ##info bit 6
# data = {
#     1: {
#         "MC-NCJT": {"ber": 0.004945090964988385, "mod": 6, "rate": 0.5},
#         "CJT (Ideal scenario)": {"ber": 0.0, "mod": 4, "rate": 0.5},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 4, "rate": 0.5},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 4, "rate": 0.5},
#     },
#     4: {
#         "MC-NCJT": {"ber": 0.00016999421296294667, "mod": 6, "rate": 0.5},
#         "CJT (Ideal scenario)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
#     },
#     7: {
#         "MC-NCJT": {"ber": 1.1302806712962962e-06, "mod": 6, "rate": 0.5},
#         "CJT (Ideal scenario)": {"ber": 0.0, "mod": 2, "rate": 0.33},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 2, "rate": 0.33},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 2, "rate": 0.33},
#     }
# }

#info bit 8
data = {
    2: {
        "MC-NCJT": {"ber": 0.00669945610894092, "mod": 8, "rate": 0.5},
        "CJT (Ideal scenario)": {"ber": 0.0, "mod": 4, "rate": 0.5},
        "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 4, "rate": 0.5},
        "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 4, "rate": 0.5},
    },
    4: {
        "MC-NCJT": {"ber": 0.0031494140624999598, "mod": 8, "rate": 0.5},
        "CJT (Ideal scenario)": {"ber": 0.0, "mod": 4, "rate": 0.33},
        "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 4, "rate": 0.33},
        "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 4, "rate": 0.33},
        
    },
    6: {
        "MC-NCJT": {"ber": 0.0009767320421006445, "mod": 8, "rate": 0.5},
        "CJT (Ideal scenario)": {"ber": 0.0, "mod": 2, "rate": 0.5},
        "CJT (Perfect CSI w. Prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
        "CJT (Estimated CSI w. prediction)": {"ber": 0.0, "mod": 2, "rate": 0.5},
    }
}


########## UnCoded BER
#Total bit 6
# data = {
#     1: {
#         "SC-NCJT": {"ber": 0.0006027620602277069, "mod": 6},
#         "CJT (Ideal scenario)": {"ber": 4.478064373897707e-06, "mod": 2},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 9.070950911228688e-06, "mod": 2},
#         "CJT (Estimated CSI w. prediction)": {"ber": 2.9279651675485007e-05, "mod": 2},
#     }
# }


# #Total bit 8
# data = {
#     2: {
#         "SC-NCJT": {"ber": 0.00640835730858093, "mod": 8},
#         "MC-NCJT": {"ber": 0.00233976576063362, "mod": 4},
#         "CJT (Ideal scenario)": {"ber": 3.875248015873016e-06, "mod": 2},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 2.204585537918871e-05, "mod": 2},
#         "CJT (Estimated CSI w. prediction)": {"ber": 6.984058090828924e-05, "mod": 2},
#     }
# }


# #Total bit 12
# data = {
#     1: {
#         "MC-NCJT": {"ber": 0.04904841670283559, "mod": 6},
#         "CJT (Ideal scenario)": {"ber": 0.005091559193121693, "mod": 4},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.007164960409318048, "mod": 4},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.00949602256025867, "mod": 4},
#     },
#     4: {
#         "MC-NCJT": {"ber": 0.013612309208622619, "mod": 6},
#         "CJT (Ideal scenario)": {"ber": 1.5041703409758965e-05, "mod": 2},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.00015478027630805405, "mod": 2},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.000389361956202234, "mod": 2},
#     }
# }


# #Total bit 16
# data = {
#     2: {
#         "MC-NCJT": {"ber": 0.07755813598632808, "mod": 8},
#         "CJT (Ideal scenario)": {"ber": 0.0041065572641093475, "mod": 4},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.007869035562720459, "mod": 4},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.011349180514219575, "mod": 4},
#     },
#     6: {
#         "MC-NCJT": {"ber": 0.04648628234863278, "mod": 8},
#         "CJT (Ideal scenario)": {"ber": 4.460841049382715e-05, "mod": 2},
#         "CJT (Perfect CSI w. Prediction)": {"ber": 0.0003726266258818342, "mod": 2},
#         "CJT (Estimated CSI w. prediction)": {"ber": 0.0010093298748897707, "mod": 2},
#     }
# }

#"SC-NCJT", "MC-NCJT",
#"tab:blue", "tab:green",
scenarios = ["MC-NCJT", "CJT (Ideal scenario)", "CJT (Perfect CSI w. Prediction)", "CJT (Estimated CSI w. prediction)"]
colors = ["tab:green", "tab:orange", "tab:red", "tab:purple"]

n_rx = len(data)
n_scen = len(scenarios)

gap = 2.0  # controls spacing between groups
group_width = n_scen + gap
x = np.arange(n_rx) * group_width

fig, ax = plt.subplots(figsize=(14,5))

# Plot bars + annotate
for i, scen in enumerate(scenarios):
    xpos = x + i
    yvals = [data[rx][scen]["ber"] for rx in data]
    bars = ax.bar(xpos, yvals, color=colors[i], width=0.8, label=scen)
    
    # Annotate BER on top of each bar
    for rect, val in zip(bars, yvals):
        if val > 0:
            ax.text(rect.get_x() + rect.get_width()/2, val,
                    f"{val:.1e}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    

# Y axis
ax.set_yscale("log")
ax.set_ylabel("LDPC coded BER")
ax.set_title("CJT vs NCJT for 10 km/h (Total info bits per RE over all streams = 8)")

# First-level labels: (mod, rate) for each bar
all_xpos = []
tick_labels = []
for rx in data:
    for scen in scenarios:
        all_xpos.append(0)  # placeholder, we’ll override
        tick_labels.append(f"mod={data[rx][scen]['mod']}\nrate={data[rx][scen]['rate']}")
        # tick_labels.append(f"mod={data[rx][scen]['mod']}")

# Assign xticks correctly
ax.set_xticks(np.concatenate([x + np.arange(n_scen) for x in x]))
ax.set_xticklabels(tick_labels)

# Second-level unified labels: Rx UEs
group_centers = x + (n_scen-1)/2
for gx, rx in zip(group_centers, data.keys()):
    ax.text(gx, ax.get_ylim()[0]*.7, f"Rx UEs={rx}",
            ha="center", va="top", fontsize=11, fontweight="bold")

# Legend
ax.legend()
# Place legend to the right middle outside the plot
ax.legend(loc="upper right")

plt.tight_layout()

plt.savefig("coded_ber_8.png", bbox_inches="tight")
