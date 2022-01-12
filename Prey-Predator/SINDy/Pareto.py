import numpy as np
import matplotlib.pyplot as plt
import plottools
import os
from ModelSelection import ModelSelection

# Read data
ms = ModelSelection()
optimizer_method = ms.read_pareto_curve()

# Generate Pareto curve
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
ax.plot(ms.k[ms.best_AICc_model], ms.SSE[ms.best_AICc_model], "k*", alpha = 0.5, markersize = 15, label = "Best model")
ax.plot(np.delete(ms.k, ms.best_AICc_model), np.delete(ms.SSE, ms.best_AICc_model), "ko", alpha = 0.5, markersize = 10, label = "Other identified models")
ax.set(xlabel = "Model complexity", ylabel = "Error", yscale = "log",
	# title = optimizer_method + " - Pareto curve",
	xticks = range(0, int(np.amax(ms.k))+3, 2)
)
ax.legend()

model_id = 0
for i, j in zip(ms.k, ms.SSE):
	model_id += 1
	ax.annotate(str(model_id), xy = (i, j), xytext = (10, 0), textcoords = 'offset points')

# ax_zoom = plottools.zoom_axes(fig, ax, [2.5, 12.5], [0.0, 12.0], [10.0, 20.0], [800.0, 1200.0])
# ax_zoom.plot(ms.k, ms.SSE, "bo", alpha = 0.6, markersize = 6)
# ax_zoom.plot(ms.k[ms.best_AICc_model], ms.SSE[ms.best_AICc_model], "ro", alpha = 0.6, markersize = 6)

# model_id = 0
# for i, j in zip(ms.k, ms.SSE):
#     model_id += 1
#     ax_zoom.annotate(str(model_id), xy = (i, j), xytext = (5, 0), textcoords = 'offset points')

# fig.show()
plt.savefig(os.path.join("output", "pareto.png"), bbox_inches = 'tight')
plt.close()
