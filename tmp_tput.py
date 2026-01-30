import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# RUs
RUs = [4, 6, 10]

# Throughput #done sc,mc,idealcjt
sc_ncjt_tp = [24, 24, 24]   
mc_ncjt_tp = [33, 34.5, 35]   
cjt_est_pred_tp = [25, 32, 47]

def plot_no_nan(x, y, **kwargs):
    x_valid = [xi for xi, yi in zip(x, y) if not np.isnan(yi)]
    y_valid = [yi for yi in y if not np.isnan(yi)]
    plt.plot(x_valid, y_valid, **kwargs)

plt.figure(figsize=(10,7))
plt.plot(RUs, sc_ncjt_tp, marker='s', color='red', label="SC-NCJT")
plt.plot(RUs, mc_ncjt_tp, marker='o', color='red', label="MC-NCJT")
plt.plot(RUs, cjt_est_pred_tp, marker='>', color='blue', label="CJT")

# === Annotations for 3 scenarios ===

# MC-NCJT
# for x, y in zip(rx_ues, mc_ncjt_tp):
#     if not np.isnan(y) and x in mc_ncjt_configs:
#         plt.annotate(mc_ncjt_configs[x],
#                      (x, y),
#                      textcoords="offset points",
#                      xytext=(0,8),
#                      ha='center', fontsize=12, color="black")

# # Ideal CJT with delay
# for x, y in zip(rx_ues, cjt_ideal_delay_tp):
#     if not np.isnan(y) and x in delay_cjt_configs:
#         plt.annotate(delay_cjt_configs[x],
#                      (x, y),
#                      textcoords="offset points",
#                      xytext=(0,8),
#                      ha='center', fontsize=8, color="darkred")

# # Est. CSI w. Prediction (CJT)
# for x, y in zip(rx_ues, cjt_est_pred_tp):
#     if not np.isnan(y) and x in est_pred_cjt_configs:
#         plt.annotate(est_pred_cjt_configs[x],
#                      (x, y),
#                      textcoords="offset points",
#                      xytext=(0,8),
#                      ha='center', fontsize=8, color="black")

# # Axes setup
# ax = plt.gca()
# # ax.set_yscale("log")

# # Major ticks at decades, minor ticks in between
# ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None, numticks=10))
# ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, 
#                                              subs=np.arange(2, 10) * 0.1,  # e.g. 2e-1, 3e-1, ... 9e-1
#                                              numticks=10))

# plt.yscale("log")
plt.xticks(RUs)
plt.tick_params(axis='both', which='both', labelsize=14)
plt.xlabel("Number of DU+RUs")
plt.ylabel("Throughput (Mbps)")
plt.title("Throughput Comparison: NCJT vs CJT for 4 UEs")
plt.grid(axis="y", linestyle="--", alpha=0.7, which="both")
plt.legend()

plt.tight_layout()
plt.savefig("throughput_comparison.png", dpi=300, bbox_inches="tight")
