import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/network-stats-plus-other.csv")
boxplot = data.boxplot()
plt.show()

stats = pd.DataFrame()

stats["Mean"] = data.mean().round(2)
stats["STD"] = data.std().round(2)
stats["Median"] = data.median()
stats["Min"] = data.min()
stats["Max"] = data.max()



print(stats.to_latex())
print(data.to_latex())
