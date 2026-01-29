from sklearn.neighbors import KNeighborsClassifier
x_train = [[0], [1], [2],[3]]#训练集的特征
y_train = [0, 0, 1, 1]#训练集的标签
x_test = [[5]]#测试集特征
estimator = KNeighborsClassifier(n_neighbors=2)#创建模型对象
estimator.fit(x_train, y_train)#模型训练
y_pred = estimator.predict(x_test)#模型预测
print(y_pred)#k领近算法
