# Neural-Network-Colorization

## Versões

-1.0: Utiliza-se redes neurais para predizer o canal Cr e Cb a partir do Y.

-2.0: Utiliza-se o Self Organized Maps para quantizar a quantidade de cores e por sua vez diminuir o número de targets. Em seguida, utiliza-se uma rede neural para predizer os canais U e V quantizados a partir do L.

-3.0: Utiliza-se o Self Organized Maps para quantizar a quantidade de cores e em seguida cria-se uma rede neural diferente para cada um dos clusters definidos pelo SOM. Nesta versão, classificaremos cada pixel com o classificador do cluster do qual ele faz parte.
