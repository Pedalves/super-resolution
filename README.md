# super-resolution

Este projeto visa o desenvolvimento de um sistema de apoio à experimentação da super resolução de dados sísmicos. 

![super resolution](docs/imgs/sr-seismic.png)

## Organização

`_data/`: Arquivos estáticos, como log, dataset e pesos dos modelos

`dataset/`: Módulo responsavel pelo tratamento e carregamento do dataset

`docs/`: Arquivos de documentação

`ipynb/`: Sandbox com os Jupyter notebooks utilizados durante o desenvolvimento do projeto. 
Alguns destes notebooks podem estar descontinuados devido ao desenvovlimento e alterações dos moódulos deste projeto

`leaderboard/`: desenvolvimento do leaderboard

`models/`: Desenvolvimento das redes neurais e os demais modelos de super resolução

`super_resolution/`: Módulo principal do projeto, contendo as função com o pipeline de experimentação 

`tasks/`: Funções baseadas na biblioteca Invoke para serem rodadas por linha de comando

`tests/`: Testes unitários

`utils/`: Funções auxiliares   

## Documentation

a [documentação](docs/README.md) deste projeto apresenta os dados necessarios e os módulos `dataset`, `models`, 
`super_resolution`, `tasks` e `utils`.

of this project covers the required third-party data and an overview of the pulses, trips and scores 
modules. The pulses module is responsible for estimating pulse CEP and risk region and computing the history of driver's overnight 
CEP. The trips module computes a set of features for each driver trip. The scores module computes the five scores
developed in this project - schedule, circulation, scheduled circulation, driver signature and overall scores.


## Desenvolvimento

Este projeto e desenvolvido utilizando `python 3.7.4`. É recomendado usar o [pyenv](https://github.com/pyenv/pyenv)
como gerenciador de versão python. Ele pode ser facilmente instalado com [pyenv installer](https://github.com/pyenv/pyenv-installer).

### Clone

Para clonar esse repositorio, execute o seguinte comando:

```
git clone https://github.com/Pedalves/super-resolution.git
```

- via ssh(**recomendado**):
```
git clone git@github.com:Pedalves/super-resolution.git
```

Observe que você deve ter uma chave SSH com acesso ao repositório do projeto.

### Build

Pipenv é a ferramenta de gerenciamento de pacotes Python oficialmente recomendada. 
Instale o `pipenv` conforme descrito em seu [repositoria oficial](https://github.com/pypa/pipenv#installation) and execute the following command.

```
pipenv install
``` 
