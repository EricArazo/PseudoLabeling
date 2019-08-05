import matplotlib.pyplot as plt


def save_fig(dst_folder):
    log_file = open(dst_folder + '/train.log', 'r')
    top1_val_ac = []
    top5_val_ac = []
    train_ac = []
    train_loss = []

    for line in log_file.readlines():
        line = line.strip().split()
        if len(line) < 14:
            continue
        if 'loss' in line[2]:
            str = line[3]
            train_loss.append(float(str.split(',')[0]))
        if 'train_ac' in line[4]:
            str = line[5]
            train_ac.append(float(str.split(',')[0]))
        if 'top5_val' in line[6]:
            str = line[7]
            top5_val_ac.append(float(str.split(',')[0]))
        if 'top1_val' in line[8]:
            str = line[9]
            top1_val_ac.append(float(str.split(',')[0]))

        plt.figure('result')
        plt.subplot(211)
        plt.plot(top1_val_ac)
        plt.plot(top5_val_ac)
        plt.legend([top1_val_ac, top5_val_ac],  ['top1', 'top5'], loc='upper right')
        plt.grid(True)
        plt.subplot(212)
        plt.plot(train_loss)
        plt.legend(['train_loss'], loc='upper right')
        plt.grid(True)
        plt.savefig(dst_folder + '/result.png')