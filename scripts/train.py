
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD, RMSprop,Adam
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense,Flatten, Dropout, Conv2D, MaxPooling2D

train = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_url = "D:/K22_DUT/HK5/PBL4/CNN/data/processed/train"

validation = ImageDataGenerator(rescale=1./255)
validation_url = "D:/K22_DUT/HK5/PBL4/CNN/data/processed/val"
     

train_dataset = train.flow_from_directory(train_url,
                                          target_size = (150,150),
                                          batch_size = 128,
                                          class_mode = 'categorical')
validation_dataset = validation.flow_from_directory(validation_url,
                                                    target_size = (150,150),
                                                    batch_size = 128,
                                                    class_mode = 'categorical')

train_dataset.class_indices

model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same',input_shape=(150,150,3)))
model.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(Conv2D(256,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_initializer='he_uniform',))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))

model.summary()



opt = SGD(learning_rate=0.001, momentum=0.9)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]
history = model.fit(train_dataset,batch_size=128,epochs=50,verbose=1,validation_data=validation_dataset,callbacks=callbacks)

model.save("D:/K22_DUT/HK5/PBL4/CNN/checkpoints/checkpoint-7/model-7.h5")

model_Final = load_model("D:/K22_DUT/HK5/PBL4/CNN/checkpoints/checkpoint-7/model-7.h5")
     

score=model_Final.evaluate(validation_dataset,verbose=1)
print('Sai số: ',score[0])
print('Độ chính xác: ',score[1])

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylabel('Accuracy / Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()