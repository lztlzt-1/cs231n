import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c,   where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]  # N
  num_classes = W.shape[1]  # C
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # print(y)
  # W(D, C)  X(N, D)
  for i in xrange(num_train):
    score = X[i].dot(W)
    score -= np.max(score)  # 提高计算中的数值稳定性
    correct_score = score[y[i]]  # 取分类正确的评分值
    exp_sum = np.sum(np.exp(score))
    loss += np.log(exp_sum) - correct_score
    for j in xrange(num_classes):

      if j == y[i]:
        dW[:, j] += np.exp(score[j]) / exp_sum * X[i] - X[i]
      else:
        dW[:, j] += np.exp(score[j]) / exp_sum * X[i]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)#参数调节
  dW /= num_train
  dW += reg * W#W是权重
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores = scores - np.max(scores, 1, keepdims=True)
  scores_exp = np.exp(scores)
  sum_s = np.sum(scores_exp, 1, keepdims=True)
  p = scores_exp / sum_s
  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(p - ind)

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += W * 2 * reg

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

