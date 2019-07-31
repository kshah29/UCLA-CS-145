import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('GMM.csv', header=None)
df.plot(kind='scatter', x=0, y=1, c=2, colormap='rainbow', colorbar=None)

plt.show()
