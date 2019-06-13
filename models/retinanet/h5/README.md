I pesi di **BACKEND** sono scaricati in automatico.

Sono disponibili vari pesi a seconda del modello ResNet applicato (50, 101, 152)

I pesi di **BASE** del modello allenato su COCO (80 classi) sono scaricabili da qui: 

https://github.com/fizyr/keras-retinanet/releases

Al momento sono disponibili solo quelli per RESNET50.

Il peso va rinominato in `base_{}.h5` dove tra `{}` va indicato il modello di backend (`resnet50`, `resnet101`, `resnet152`) 

Il file `pretrained.h5` se esiste è quello che viene caricato all'inizio del training come partenza (è impostato sulle classi indicate in `configRetinaNet.json`).

Il file `result.h5` se esiste è quello creato alla fine dell'allenamento.

Il file `chkpnt_best.h5` se esiste è quello creato durante l'allenamento come checkpoint migliore. Può essere rinominato in `pretrained.h5` se si interrompe l'allenamento e successivamente si vuole ripartire da quel punto.

Tutti i nomi ed i path dei file sono comunque configurabili da `configRetinaNet.json`.


