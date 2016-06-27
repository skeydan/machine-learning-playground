from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#2.5 * x1 - x2 = 0
#x2 = 2.5 * x1

'''
def score(x): return 2.5*x

x1s_neg = [0.5,1,1.2,1,1.7,2]
x2s_neg = [4.1,6,3.9,5,6.2,6.4]

x1s_pos = [2,3,3,4,2.5,2]
x2s_pos = [1,2,6,5.5,4.5,3.5]

x = np.arange(0,8,0.01)
y = map(score,x)

plt.figure()
plt.xkcd()
plt.scatter(x1s_neg, x2s_neg, color='red', marker='x', label='negative tweets')
plt.scatter(x1s_pos, x2s_pos, color='green', marker='+', label='positive tweets')
plt.plot(x,y)

plt.legend(loc='upper right')
plt.xlabel('awesome')
plt.ylabel('disappointed')
plt.xlim(0,7)
plt.xticks(np.arange(7))
plt.ylim(0,7)
plt.yticks(np.arange(7))
plt.show()

'''

x1 = np.random.uniform(0,10,30)
y1 = np.random.uniform(7,12,30)

x2 = np.random.uniform(5,8,20)
y2 = np.random.uniform(16,22,20)

x3 = np.random.uniform(13,20,25)
y3 = np.random.uniform(0,5,25)

plt.figure()
plt.xkcd()
plt.scatter(x1, y1, color='red', marker='x', label='some movies')
plt.scatter(x2, y2, color='green', marker='+', label='some other movies')
plt.scatter(x3, y3, color='blue', marker='o', label='yet other movies')

plt.legend(loc='upper right')
plt.xlabel('genre: drama')
plt.ylabel('genre: politics')
plt.xlim(0,25)
plt.xticks(np.arange(25))
plt.ylim(0,25)
plt.yticks(np.arange(25))
plt.show()


x1 = np.random.uniform(2,10,50)
y1 = np.random.uniform(7,12,50)

x2 = np.random.uniform(8,22,70)
y2 = np.random.uniform(16,22,70)

x3 = np.random.uniform(9,24,90)
y3 = np.random.uniform(0,5,90)

plt.figure()
plt.xkcd()
plt.scatter(x1, y1, color='magenta', marker='x', label='some users')
plt.scatter(x2, y2, color='cyan', marker='+', label='some other users')
plt.scatter(x3, y3, color='orange', marker='o', label='yet other users')

plt.legend(loc='center right')
plt.xlabel('interest in drama')
plt.ylabel('interest in politics')
plt.xlim(0,25)
plt.xticks(np.arange(25))
plt.ylim(0,25)
plt.yticks(np.arange(25))
plt.show()






