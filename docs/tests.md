# Tests

O testes unitários desse projeto foram desenvolvidos com o apoio da biblioteca [PyTest](https://docs.pytest.org/en/stable/).

O log da execução dos testes se encontra no arquivo ``_data/log/test.log``.

## Dataset 

* test_load_model_shape: Confere se as dimensões dos poços carregados estão como o esperado.

* test_remove_water: Confere se a remoção da fatia do modelo correspondente a água esta sendo feita corretamente.
a imagens correspondente a água possuem desvio padrão zero, com isso é conferido se existe alguma imagem no dataset com este valor.

* test_get_rgb_data: Confere se a função ``get_cmap()`` está retornando o vetor com 3 canais.

* test_img_size: Confere as dimensões das imagens particionadas do modelo e de suas versões após o down sampling.

* test_normalize: Confere se a normalização está sendo feita corretamente.

## Métricas

* test_mse: Compara se o resultado retornado do MSE de uma vetor com o valor esperado.
  
* test_mse_all_equal: Calcula o MSE entre dois vetores iguais, o valor esperado do erro neste caso é zero.
    
* test_psnr: Compara se o resultado retornado do PSNR de uma vetor com o valor esperado.
  
* test_psnr_all_equal: Calcula o PSNR entre dois vetores iguais, o valor esperado do erro neste caso é zero.

## Modelos

* test_neural_networks_super_class: Confere se os modelos de redes neurais desenvolvidos herdam da 
super classe ``NeuralNetwork``.

* test_neural_network_name: Confere se o nome do modelo está de acordo com a nomeclatura esperada.

* test_save_neural_network: Treina e salva um modelo. Em seguida carrega este modelo e confere se as métricas de
 MSE e PSNR continuam as mesmas.