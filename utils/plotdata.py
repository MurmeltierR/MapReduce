import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import dataframe_image as dfi

df = pd.read_csv("./test_neu3.csv", header=None).head(3)
df.dfi.export('df.png')
# ax = plt.subplot() #
# # ax.xaxis.set_visible(False)  # hide the x axis
# # ax.yaxis.set_visible(False)  # hide the y axis

# table(ax, df)  # where df is your data frame

# plt.savefig('mytable.png')