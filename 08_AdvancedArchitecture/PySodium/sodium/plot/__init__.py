import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.style.use("dark_background")


def plot_metrics(train_metric, test_metric):
    (train_losses, train_acc) = train_metric
    (test_losses, test_acc) = test_metric

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics')
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()


def plot_misclassification(misclassified):
    print('Total Misclassifications : {}'.format(len(misclassified)))
    num_images = 25
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle('Misclassifications')
    for idx, (image, pred, target) in enumerate(misclassified[:num_images]):
        image, pred, target = image.cpu().numpy(), pred.cpu(), target.cpu()
        ax = fig.add_subplot(5, 5, idx+1)
        ax.axis('off')
        ax.set_title('target {}\npred {}'.format(
            target.item(), pred.item()), fontsize=12)
        ax.imshow(image.squeeze())
    plt.show()
