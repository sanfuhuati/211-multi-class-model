import matplotlib.pyplot as plt

def plot_precision_recall_vs_iterations(precision_list, recall_list, f1_list, hamming_list,title='Precision and Recall vs Iterations'):

    iterations = range(1, len(precision_list) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(iterations, precision_list, marker='o', label='Precision', color='blue', linestyle='-', linewidth=1)

    plt.plot(iterations, recall_list, marker='s', label='Recall', color='red', linestyle='--', linewidth=1)
    plt.plot(iterations, f1_list, marker='s', label='f1', color='green', linestyle='--', linewidth=1)
    plt.plot(iterations, hamming_list, marker='s', label='hamming', color='yellow', linestyle='--', linewidth=1)

    plt.title(title, fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.xlim(1, len(precision_list))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_pr_curve(precisions, recalls):
    plt.figure(figsize=(8, 6))
    for precision, recall in zip(precisions, recalls):
        plt.plot(recall, precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

def plot_precision_recall(precision, recall, labels, title='Precision and Recall for Each Label'):

    plt.figure(figsize=(10, 6))
    plt.plot(labels, precision, marker='o', label='Precision', color='blue', linestyle='-', linewidth=2)
    plt.plot(labels, recall, marker='s', label='Recall', color='red', linestyle='--', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel('Labels', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()



import pandas as pd
import os

# 定义 CSV 文件路径
csv_file = 'metrics.csv'

# 每次迭代追加数据
def append_metrics(precision, recall, f1, hamming_accuracy):
    data = {
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'Hamming_Accuracy': [hamming_accuracy]
    }
    df = pd.DataFrame(data)
    
    # 如果文件存在，追加模式；否则创建新的文件
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

# 示例调用
append_metrics(0.9, 0.8, 0.85, 0.95)
append_metrics(0.85, 0.75, 0.8, 0.92)
append_metrics(0.8, 0.7, 0.75, 0.89)
