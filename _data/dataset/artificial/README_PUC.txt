

Tem-se os modelos de velocidades e as imagens migradas orindas da modelagem
sismica e imageamento via RTM (Reverse Time Migration) de tais modelos.

OBS:	- As imagens podem/devem ser tratadas (filtros low-cut) a depender do objetivo
pretendido.
	- Alem da imagem original, tem-se um exemplo de pos-processamento
	  (filtro low-cut) que pode ser aplicado a imagem.


DEFINICAO DAS VARIAVEIS:

n1 - (numero de amostras (linhas da imagem pixel))
n2 - (numero de tracos (colunas da imagem)
d1 e d2 - (intervalo de amostragem nas direcoes X e Y)
VMIN e VMAX - (valores maximo e minimo de velocidades, porpriedades (o valor de 4500 corresponde ao sal)


Modelo 1 - SEG/EAGE Modified -
	Arquivo de velocidades: mod_vp_05_nx7760_nz1040.bin 
	Imagem migrada: IMG1_dip_FINAL_REF_model_1_true.bin
	n1=1040 ; n2=7760 
	d1=5    ; d2=5  
	VMIN=1500 ; VMAX=4500 

Modelo 2 - Pluto -
	Arquivo de velocidades: pluto_VP_SI_02.bin 
	Imagem migrada: IMG1_dip_FINAL_REF_model_2_true.bin
	n1=1216 ; n2=6912
	d1=7.62 ; d2=7.62
	VMIN=1490 ; VMAX=4550


