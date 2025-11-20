#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from matplotlib import pyplot as plt
import numpy as np


# In[3]:


(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = datasets.mnist.load_data()


# In[4]:


print(x_train_raw.shape, y_train_raw.shape)
print(x_test_raw.shape, y_test_raw.shape)
print(x_train_raw[0])
print(y_train_raw[0])


# In[22]:


num_classes = 10
y_train = keras.utils.to_categorical(y_train_raw, num_classes)
y_test = keras.utils.to_categorical(y_test_raw, num_classes)
print(y_train_raw[0])
print(y_train[0])


# In[8]:


plt.figure()
for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.imshow(x_train_raw[i])
  plt.axis('off')
plt.show()


# In[9]:


x_train = x_train_raw.reshape(60000, 784)
x_test = x_test_raw.reshape(10000, 784)


# In[10]:


x_train = x_train.astype('float32')/225
x_test = x_test.astype('float32')/225


# In[11]:


model = keras.Sequential([
                          layers.Dense(512, activation='relu', input_dim= 784),
                          layers.Dense(256, activation='relu'),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(num_classes, activation='softmax')


])
model.summary()


# In[13]:


Optimizer = optimizers.Adam(0.001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Optimizer,
              metrics=['accuracy'])


# In[23]:


history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    verbose=1)


# In[16]:


score = model.evaluate(x_test, y_test, verbose=0)
print('perda do teste', score[0])
print('acuracia do teste', score[1])


# In[18]:


model.save('./model/final_DNN_model.h5')


# In[19]:


from tensorflow.keras.models import load_model
new_model = load_model('./model/final_DNN_model.h5')
new_model.summary()


# In[20]:


new_score = new_model.evaluate(x_test, y_test, verbose=0)
print('perda do teste', new_score[0])
print('acuracia do teste', new_score[1])


# In[25]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='treinamento')
plt.plot(history.history['val_accuracy'], label='validação')
plt.title('acuracia por Epoca')
plt.xlabel('Epocas')
plt.ylabel('Acuracias')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='treinamento')
plt.plot(history.history['val_loss'], label='validação')
plt.title('perda por Epoca')
plt.xlabel('Epocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()


# In[30]:


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

y_true  = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()


# In[31]:


y_pred_prob = new_model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

acc = accuracy_score(y_true, y_pred)
print(f'Acurácia: {acc:.4f}')

prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print(f'precisão: {prec:.4f}')
print(f'recall: {rec:.4f}')
print(f'f1-score (macro): {f1:.4f}')

kappa = cohen_kappa_score(y_true, y_pred)
print(f'Kappa: {kappa:.4f}')

print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred))

n_classes = y_pred_prob.shape[1]
y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC Multiclasse')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


# In[34]:


model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1) , padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)))


model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(keras.layers.Conv2D(filters=64,
                              strides=(1, 1),
                              kernel_size=(3, 3),
                              padding='same',
                              activation=tf.nn.relu))


model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))


model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=128, activation=tf.nn.relu))


model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.summary()


# In[36]:


Optimizer = optimizers.Adam(0.001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Optimizer,
              metrics=['accuracy'])

x_test_cnn = x_test_raw.reshape(10000, 28, 28, 1)
x_test_cnn = x_test_cnn.astype('float32') / 225

x_test_loss, test_acc=model.evaluate(x=x_test_cnn,y=y_test)
print('Acuracia do Teste da CNN:%.2f'%(test_acc*100))
print('Perda do teste da CNN:%2f'%x_test_loss)


# In[49]:


from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax

custom_objects = {'softmax_v2': softmax}
new_model = load_model('./model/final_CNN_model.h5', custom_objects=custom_objects)
new_model.summary()


# In[54]:


history_cnn = model.fit(x_train.reshape(60000, 28, 28, 1), y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(x_test_cnn, y_test),
                    verbose=1)


# In[51]:


model.save('./model/final_CNN_model.h5')


# In[52]:


from keras.layers import Activation

model = keras.models.load_model(
   './model/final_CNN_model.h5',
    custom_objects={"softmax_v2": Activation("softmax")}
)


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
def res_visual(n):
  final_opt_a=np.argmax(model.predict(x_test_[0:n], axis=1))
  #print(final_opt_a)
  fig, ax = plt.subplots(nrows=int(n/5), ncols=5)
  ax = ax.flatten()
  print('resultados da Previsão das 20 primeiras imagens do Teste')
  for i in range(n):
    print(final_opt_a[i], end=',')
    if int((i+5)%5) == 0:
      print('\t')
  img = x_test[i].reshape(28, 28)
  plt.axis("off")
  ax[i].imshow(img, cmap="gray", interpolation='nearest')
  ax[i].axis('off')
  print("primeiras 20 primeiras linhas imagens di conjunto teste")

  res_visual(30)


