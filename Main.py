import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout
from keras.utils import np_utils
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras.layers.normalization import BatchNormalization
from ann_visualizer.visualize import ann_viz

def vizualizaResultados ( cnn ) :

    plt.figure(0)
    plt.plot(cnn.history['acc'],'r')
    plt.plot(cnn.history['val_acc'],'g')
    plt.xticks(np.arange(0, 101, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])
    plt.grid(True)

    plt.figure(0)
    plt.plot(cnn.history['loss'],'r')
    plt.plot(cnn.history['val_loss'],'g')
    plt.xticks(np.arange(0, 101, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])
    plt.grid(True)

    plt.show()

    return

def salvaRede ( cNN ) :
    """

    Função que salva os pesos e a estrutura neural da rede
    em um arquivo json

    """

    cNNJson = cNN.to_json()

    with open("Cnn.json", "w") as jsonFile :

        jsonFile.write ( cNNJson )

    cNN.save_weights("CnnWeights.h5")

    return

def treinaCNN ( alturaImagem, larguraImagem ) :
    """
    Função que treina a rede neural
    convolucional com a base de dados
    de treinamento

    """

    print("------ Início do treinamento -------\n")

    cNN = Sequential ( )

    # o número de kernels vem da forma n = 32*i
    # onde i você decide a quantidade de bytes

    # camada de convolução
    cNN.add ( Conv2D  (
        filters = 32,
        kernel_size = ( 3, 3 ),
        input_shape = ( larguraImagem, alturaImagem, 3 ),
        activation = "relu"
    ) )

    cNN.add ( Conv2D  (
        filters = 32,
        kernel_size = ( 3, 3 ),
        input_shape = ( larguraImagem, alturaImagem, 3 ),
        activation = "relu"
    ) )

    # normalização dos dados após a 1 camada de convolução
    cNN.add ( BatchNormalization (  ) )

    # camada de pooling
    cNN.add ( MaxPooling2D (
        pool_size = ( 2, 2 )
    ) )

    cNN.add ( Dropout ( 0.25 ) )

    # 2 camada da rede neural
    cNN.add ( Conv2D (
         filters = 64,
         kernel_size = ( 3, 3 ),
         input_shape = ( larguraImagem, alturaImagem, 3 ),
         activation = "relu"
    ) )

    cNN.add ( Conv2D (
         filters = 64,
         kernel_size = ( 3, 3 ),
         input_shape = ( larguraImagem, alturaImagem, 3 ),
         activation = "relu"
    ) )

    cNN.add ( BatchNormalization (  ) )

    cNN.add ( MaxPooling2D (
        pool_size = ( 2, 2 )
    ) )

    # camada de Flatten
    cNN.add ( Flatten ( ) )

    # montando a rede neural de camada densa de duas camadas escondidas
    cNN.add ( Dense (
        units = 128,
        activation = "relu"
     ) )

    cNN.add ( Dropout ( 0.5 ) )

    cNN.add ( Dense (
        units = 64,
        activation = "relu"
    ) )

    #cNN.add ( Dropout ( 0.0 ) )

    # camada saída
    cNN.add ( Dense (
         units = 10,
         activation = "softmax"
     ) )

    cNN.compile (
         loss = "categorical_crossentropy",
         optimizer = "adam",
         metrics = ["accuracy"]
     )

    #ann_viz ( cNN, title = "Rede Convolucional" )

    ##

    return cNN


def preImageProcessing ( dadoEntrada, dadoEntradaTeste ) :
    """
    Função que faz o pré - processamento de todas
    as imagens do dataset de treino e teste

    """

    previsaoTreinamento = dadoEntrada.astype ("float32")
    previsaoTeste = dadoEntradaTeste.astype ("float32")

    previsaoTreinamento /= 255
    previsaoTeste /= 255

    return previsaoTreinamento, previsaoTeste

def main (  ) :

    seed = 10

    np.random.seed ( seed )

    ( dadoEntrada, dadoSaida ), ( dadoEntradaTeste, dadoSaidaTeste ) = cifar10.load_data (  )

    previsaoTreinamento, previsaoTeste = preImageProcessing (
        dadoEntrada = dadoEntrada,
        dadoEntradaTeste = dadoEntradaTeste
    )

    dummyRespTreinamento = np_utils.to_categorical ( dadoSaida, 10 ) # conversão dos dados categóricos de treinamento
    dummyRespTeste = np_utils.to_categorical ( dadoSaidaTeste, 10 )

    cNN = treinaCNN (
        alturaImagem = len ( dadoEntrada[:][0] ),
        larguraImagem = len ( dadoEntrada[0][:] )
    )

    sequential_model_to_ascii_printout ( cNN )

    cNN.fit ( previsaoTreinamento, dummyRespTreinamento,
              batch_size = 32,
              epochs = 20,
              validation_data = ( previsaoTeste, dummyRespTeste )
    )

    resultado = cNN.evaluate (
        previsaoTeste,
        dummyRespTeste
    )

    print(cnn.history['acc'])


    vizualizaResultados ( cNN )

    print("Accuracy: %.2f%%" % (resultado[1]*100))

if __name__ == '__main__':

    main (  )
