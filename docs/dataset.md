# Dataset

Este módulo é responsavel pelo carregamento e processamento do dataset. 

Durante o prcessamento é construido o conjunto de trino através da técnica de down sampling. Ademais, é feita a quebra 
da imagem em conjuntos ``n`` imagens de tamanho ``img_size x img_size`` menores e a partição do conjunto de treino, teste e validação na 
proporção de 50%, 20% e 30% respectivamente.

## Artificial Dataset Reader

Classe desenvolvida para uso dos dados artificiais fornecidos pela Petrobras. Para mais detalhes sobre este dataset
confira em [Requisitos de dados](data_requirements.md).

O pré processamento feito neste conjunto de dados consiste na retirada de regiões de água, 
a retirada de regiões duplicadas e, opcionalmente, a conversão dos dados em uma imagem com 3 canais RGB 
através do uso de colormaps.

### Execução
``` python
from dataset.artificial_dataset import ArtificialDatasetReader

dr = ArtificialDatasetReader(dataset_path)
x_train, y_train, x_val, y_val, x_test, y_test = dr.get_dataset(scale, img_size, as_cmap)
```

* __init\__: Construtor da classe.
    * dataset_path: Caminho para a pasta contendo o dataset. O valor default é ``_data/dataset/artificial``.

* get_dataset: Retorna o conjunto de dados separado entre treino, validação e teste.
    * scale: A escala de resolução desejada. O valor default é ``2``.
    * img_size: O tamanho das imagens. O valor default é ``64``.
    * as_cmap: Booleano representado se a imagem será convertida em uma imagem com 3 canais RGB. O valor default é ``False``.

* load_dataset: Carrega os dados do poço em um numpy array.
    * **height**: Altura do poço.
    * **length**: Largura do poço.
    * **vel_path**: Caminho para a pasta contendo o arquivo binário da velocidade.
    * **well_path**: Caminho para a pasta contendo o arquivo binário do poço.
    * **edges**: Tupla contendo o limite inferior e superior dos limites das bordas.
    * normalize: Booleano para a normalização dos dados. O valor default é ``False``.

* _normalize: Normalização min max.
    * **well**: Array a ser normalizado.
    
* get_cmap: Converte um array em uma imagem com 3 canais RGB através do uso de colormaps
    * **array**: Array a ser convertido.