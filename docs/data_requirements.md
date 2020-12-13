# Requisitos de dados

O projeto depende de um conjunto de dados externo proveniente da petrobras. Existem 2 modelos sísmicos
 necessários para o funcionamento do sistema:
 
Definição das variaveis:
```
n1 - (número de amostras (linhas da imagem pixel))
n2 - (número de tracos (colunas da imagem)
d1 e d2 - (intervalo de amostragem nas direções X e Y)
VMIN e VMAX - (valores máximo e mínimo de velocidades, porpriedades (o valor de 4500 corresponde ao sal)
```

- Modelo 1 - SEG/EAGE Modified -
	- Arquivo de velocidades: mod_vp_05_nx7760_nz1040.bin 
	- Imagem migrada: IMG1_dip_FINAL_REF_model_1_true.bin
	- n1=1040 ; n2=7760 
	- d1=5    ; d2=5  
	- VMIN=1500 ; VMAX=4500 

- Modelo 2 - Pluto -
	- Arquivo de velocidades: pluto_VP_SI_02.bin 
	- Imagem migrada: IMG1_dip_FINAL_REF_model_2_true.bin
	- n1=1216 ; n2=6912
	- d1=7.62 ; d2=7.62
	- VMIN=1490 ; VMAX=4550
