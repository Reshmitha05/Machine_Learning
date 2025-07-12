### Poker Hand Classification with a Neural Network (from Scratch)

In this project I built and trained a **neural network from scratch** to classify different poker hands. No TensorFlow. No PyTorch. Just raw matrices and a whole lot of dot products.


### What is This About?

The dataset is based on a simplified version of poker. Each sample (row) represents a poker hand (5 cards) with:
- 10 input features = 5 cards × (suit, rank)
- Suits: Clubs, Diamonds, Hearts, Spades (represented as 1-4)
- Ranks: 1 (Ace) to 13 (King)

The goal is to **predict the type of poker hand**, labeled from:
0: Nothing in hand
1: One pair
2: Two pairs
3: Three of a kind
4: Straight
5: Flush
6: Full house
7: Four of a kind
8: Straight flush
9: Royal flush

And the data set is highly imbalanced, as seen from confusion matrix.

---

I implemented a **fully connected neural network from scratch**, including:
- Custom forward and backward propagation 
- Support for **ReLU** and **Sigmoid** activations
- **Mini-batch SGD** optimizer
- **Cross-entropy loss** (can try with mse too)
- **Adaptive learning rate** (can set as zero too)
- **Training accuracy and learning curve plotting**
- **Confusion matrix** visualization

---


###  Best Model (ReLU + [64, 64] hidden layers)
Tried with [5,10,15,20] hidden layers too, but the accuracy is too low for them. Best no. of hidden layers in 2.

| Metric            | Value     |
|-------------------|-----------|
| Train Accuracy    | 98.36%    |
| Test Accuracy     | 97.01%   |
| Learning Curve    | Smooth convergence across 30 epochs |
| Confusion Matrix  | Great performance across all classes (including rare ones) |

Also tried with Sigmoid activation instead of ReLU, but the accuracy is too low (~50.09% for test data, ~49.92% for train data)

###  Sklearn Comparison

I also trained `MLPClassifier` from Scikit-learn with the same architecture:

| Metric            | Value     |
|-------------------|-----------|
| Test Accuracy     | **99.32%**  |
| Notes             | Even better accuracy, thanks to momentum, regularization, etc. |

###  Data Handling

- Manually downloaded the dataset from [UCI ML Repo](https://archive.ics.uci.edu/dataset/158/poker+hand)
- Split into training and testing sets (`poker-hand-training-true.data` and `poker-hand-testing.data`)
- No missing values, no normalization needed (categorical input)
- Tried label encoding vs one-hot (used label encoding for my main model)
- Used simple exploratory data analysis to confirm class imbalance

---

- Implementing **backpropagation** and **cross-entropy** manually gave me a much deeper intuition.
- ReLU consistently outperforms Sigmoid for hidden layers
- Adaptive learning rate improves convergence without hurting performance
- Dataset imbalance is real — majority class dominates ( confusion matrix helped visualize that)
- Scikit-learn models can serve as great baselines

---

### How to Run (if you want to...)

1. Clone the repo
2. Make sure you have `numpy`, `matplotlib`, and `sklearn` installed
3. Place the dataset files (`poker-hand-training-true.data`, `poker-hand-testing.data`) in the same directory
4. Run the notebook: `Poker_NN_Notebook.ipynb`


### Dataset Credit:

- UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/158/poker+hand

