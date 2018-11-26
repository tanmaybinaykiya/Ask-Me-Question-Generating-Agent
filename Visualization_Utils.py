import matplotlib.pyplot as plt


def plot_losses(losses):
    plt.plot(losses)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)

    plt.show()
