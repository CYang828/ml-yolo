# 准备和体验

## 机器学习中常用的包
- scikit-learn
- pandas
- numpy
- matplotlib
- ploty

### Numpy


```python
import numpy as np

# 数组转换ndarray
l = [1, 2, 3, 4]
np_l = np.array(l)
np_l
```




    array([1, 2, 3, 4])




```python
l1 = [[1,2,3], [5,4,1], [3,6,7]]
np_l1 = np.array(l1)
np_l1
```




    array([[1, 2, 3],
           [5, 4, 1],
           [3, 6, 7]])




```python
np.save('l1', l1)
np.savez('multi_l', l, l1)
```


```python
np.load('l1.npy')
```




    array([[1, 2, 3],
           [5, 4, 1],
           [3, 6, 7]])




```python
np.zeros(7)
np.zeros((3, 4))
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
np.ones(5)
```




    array([1., 1., 1., 1., 1.])




```python
np.arange(10)
np.arange(1, 11, 2)
```




    array([1, 3, 5, 7, 9])




```python
np.linspace(1, 3, 15)
```




    array([1.        , 1.14285714, 1.28571429, 1.42857143, 1.57142857,
           1.71428571, 1.85714286, 2.        , 2.14285714, 2.28571429,
           2.42857143, 2.57142857, 2.71428571, 2.85714286, 3.        ])




```python
np.random.rand(4)
```




    array([0.7711299 , 0.647307  , 0.82716935, 0.15071515])




```python
np.random.randn(3, 5)
```




    array([[-0.2367645 , -2.33515251,  1.40117218,  0.08218369, -0.79195348],
           [ 0.97489154,  1.16495338, -0.53003192, -1.21692175,  0.62282516],
           [-0.28658248, -0.59121136,  0.16285438,  1.76758292, -2.0781531 ]])




```python
# int8,int16,int32,float32
n1 = np.array([1, 2, 3], dtype=np.float32)
print(n1)
```

    [1. 2. 3.]



```python
n1.dtype
```




    dtype('float32')




```python
n1.astype(np.float16)
```




    array([1., 2., 3.], dtype=float16)




```python
names = np.array(['bob', 'joe', 'will', 'bob', 'will'])
names == 'bob'
```




    array([ True, False, False,  True, False])




```python
data = np.random.randn(5, 4)
data[names == 'bob']
```




    array([[-1.16399937,  0.8759341 , -0.13342718, -0.5107893 ],
           [-0.05942432,  1.32843359,  0.0570511 , -2.13018033]])




```python
np.arange(32).reshape((8, 4))
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
# 广播机制
na = np.arange(10)
na[0:3] = 50
na
```




    array([50, 50, 50,  3,  4,  5,  6,  7,  8,  9])




```python
# 开平方
arr = np.arange(10)
np.sqrt(arr)
```




    array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
           2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])




```python
# 内积
x = np.array([[1,2,3], [4, 5, 6]])
y = np.array([[6,12], [2,3], [12,4]])
print(x.shape, y.shape)
np.dot(x, y)
```

    (2, 3) (3, 2)





    array([[ 46,  30],
           [106,  87]])




```python
from numpy.linalg import det, eig, qr, svd
# 奇异值分解
svd(x)
```




    (array([[-0.3863177 , -0.92236578],
            [-0.92236578,  0.3863177 ]]),
     array([9.508032  , 0.77286964]),
     array([[-0.42866713, -0.56630692, -0.7039467 ],
            [ 0.80596391,  0.11238241, -0.58119908],
            [ 0.40824829, -0.81649658,  0.40824829]]))




```python
# 计算特征向量
t = np.array([[1,2,3], [4, 5, 6], [5, 9, 0]])
eig(t)
```




    (array([12.17621737, -0.57147401, -5.60474336]),
     array([[-0.29884529, -0.86936264, -0.28007282],
            [-0.70464817,  0.47413382, -0.39052644],
            [-0.64355454,  0.13930439,  0.876954  ]]))



Numpy重点的易混淆的点：
- 维度(dimensions)叫做轴(axes)，轴的个数叫rank(不是矩阵中的rank)

### Pandas

### Scikit-learn
