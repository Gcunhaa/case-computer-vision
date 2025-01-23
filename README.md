# case-computer-vision



# Code blog

Na definição do case foi dito que vocês gostariam de ter uma visão geral de como o candidato ataca o problema, a linha de raciocínio e a solução proposta. Pensando nisso vou tentar resumir passo a passo o que eu fiz pra resolver o problema proposto. 


### Contexto 23/01/2025 4:44PM

Gostaria de deixar claro que na semana que recebi o case, tive que trabalhar sabado e domingo para restruturar um projeto do zero que estava todo quebrado com criação de roadmaps e tarefas, além disso uma entrega extremamente urgente para não travar a operação da empresa com o mesmo deadline do case e a entrega de um freela. Dito isso, não tenho tempo para atacar o problema como eu gostaria de atacar, mas vou tentar fazer o melhor que eu puder.


### Analisando o problema 23/01/2024 4:48PM
Meu primeiro passo foi entender o problema e fazer algumas suposições de como resolver o problema e os diferentes cenários, meus passos iniciais foram:
- Pensei: preciso fazer uma solução que funcione em diversos cenários, levando em consideração exposição de luz, resolução de camera, angulo de visão, etc...
- Pesquisei no google um pouco sobre métodos para melhorar resolução e qualidade de imagens, cheguei em alguns papers interessantes, deixei separado para caso de tempo começar a trabalhar nisso.
- Encontrei um paper que me ajudou a entender melhor o problema e como resolver. [Studies advanced in license plate recognition](https://www.researchgate.net/publication/372823340_Studies_Advanced_in_License_Plate_Recognition)

### 24/01/2025 5:11PM
Nesse momento acabei de terminar de ler o Paper, a conclusão que eu cheguei confirma minha intuição de que a melhor solução seria dividir o problema em multiplas etapas, e é assim que vou atacar o problema. Ainda não decidir como vou atacar cada etapa, mas vou começar a pensar nisso daqui a pouco.
Etapas:
1) Preprocessamento da imagem
2) Detecção da placa
3) Segmentação dos caracteres da placa
4) Reconhecimento dos caracteres da placa

### 24/01/2025 5:22PM
Para ser sincero, fazia tempo que não fazia uma CLI então dei uma leve pesquisada em bibliotecas para me ajudar. Encontre a [Fire](https://github.com/google/python-fire/blob/master/docs/guide.md) do Google, dei uma lida na documentação, vi que é simples de implementar e decidi usar ela. Não quero gastar tempo com isso.
