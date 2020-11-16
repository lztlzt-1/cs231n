import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1  # 记住 delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i, :].T#dw存的是loss函数对应的矩阵形式
        dW[:, j] += X[i, :].T

    # 现在loss值是所有训练样例loss的总数，现在我们想要通过除以num_train来求平均值
  loss /= num_train
  dW /= num_train

  # 给loss添加正则项
  loss += reg * np.sum(W * W)

  # 计算损失函数的梯度并存储在dW中。
  # 相比较第一次那样计算loss然后计算导数，在相同时间内它可能更快的计算出loss导数
  # loss正在被计算的时候。你可能需要修改上面的一些代码来计算梯度。
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = X.dot(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores_correct = scores[np.arange(num_train), y]#
  #这一行代码的用法很巧妙，用到的原理还不是很清楚，但实现的操作是找到scores中每
  # 一行y位置的元素，最后返回scores_correct（1 by N）,
  #所以scores_correct中记录的是每一行（每一个样本）正确的分类得分结果
  scores_correct = np.reshape(scores_correct, (num_train, 1))
  #将scores_correct变成n*1
  margins = scores - scores_correct + 1
  margins = np.maximum(0, margins)#找到>0的，就是分隔线外的
  margins[np.arange(num_train), y] = 0#分类正确的标记成0
  loss += np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  margins[margins > 0] = 1
  row_sum = np.sum(margins, axis=1)  # 1 by N
  # print(row_sum)
  margins[np.arange(num_train), y] = -row_sum
  # print(margins)
  dW += np.dot(X.T, margins) / num_train + reg * W  # D by C
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
