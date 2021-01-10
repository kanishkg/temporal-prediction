import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.animation as animation
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# O1, video index from 0 to 29

data = np.load('IN_Y_15fps_1024_log.npz')
y = data['o3']
labels = data['y3']
print(y.shape)
print(labels.shape)

trial = 1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
    '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15/dev/O3',
    transforms.Compose([transforms.ToTensor(), normalize])
)

print(len(train_dataset.imgs))

fig = plt.figure(1)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

counter = 0
for i in np.arange(4*trial, 4*trial+4):

    ax1.clear()

    if labels[i] == 0:
        sty_str = 'r-'
    else:
        sty_str = 'b-'

    for j in np.arange(1, 100):

        im = plt.imread(train_dataset.imgs[100*i+j][0])
        ax1.imshow(im)
        ax1.axis('off')

        ax2.plot(np.arange(j), y[:j, i], sty_str)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 0.3)

        plt.xlabel('Time (frames)')
        plt.ylabel('Surprisal')
        plt.yticks([0, .1, .2, .3], ['0', '0.1', '0.2', '0.3'])
        plt.xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100'])

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        # ax2.yaxis.set_ticks_position('left')
        # ax2.set_ticks_position('bottom')

        print(i, j)
        counter += 1

        # FINAL CHANGES, SAVE FIGURE
        mp.rcParams['axes.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 0.75
        mp.rcParams['patch.linewidth'] = 1.15
        mp.rcParams['font.sans-serif'] = ['FreeSans']
        mp.rcParams['mathtext.fontset'] = 'cm'
        plt.savefig('./pngs/O3/' + str(trial) + '/anim' + '{:05d}'.format(counter) + '.png', bbox_inches='tight')