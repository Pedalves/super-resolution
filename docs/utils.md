# Utils

Este módulo apresenta uma série de função auxiliares para os demais módulos. A principal funcionalidade é o 
cáluclo das métricas de avaliação.

## Métricas

``` python
from utils.metrics import mse, psnr

result_mse = mse(y_val, y, verbose)
result_psnr = psnr(y_val, y, verbose)
```

* mse: Calculo do Mean Squared Error
    * **y**: Ground truth.
    * **y_pred**: Valor predito.
    * verbose: Define a verbosidade. O valor default é ``True``.
    
* psnr: Calculo do Peak Signal-to-Noise Ratio
    * **y**: Ground truth.
    * **y_pred**: Valor predito.
    * verbose: Define a verbosidade. O valor default é ``True``.
