Genetic Algorithm Based on Natural Selection Theory for Optimization Problems

colab url: https://colab.research.google.com/drive/12Y5d6MB6bgpXX9XOaon91gVQ3YBOH2S8?usp=sharing

Подробные результаты представлены в [pdf файле](https://github.com/cyansnbrst/gabonst-algorithm/blob/master/%D0%93%D0%B5%D0%BD%D0%B5%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9%20%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC.pdf)

## Вступление
Реализация основана на описанном в статье (https://www.mdpi.com/2073-8994/12/11/1758) алгоритме. С помощью него решается задача оптимизации, а именно поиск глобального минимума функции. Генетический алгоритм является одним из самых ранних и популярных алгоритмов, основанных на популяциях. Он состоит из операций селекции, кроссинговера и мутации, как и прочие эволюционные алгоритмы. В решении задач оптимизации каждый набор аргументов соответствует какой-либо хромосоме, где каждый отдельный аргумент является геном.

## Описание алгоритма
Вариация основанного в статье генетического алгоритма основана на концепции теории естественного отбора. Естественный отбор – это биологическая теория, впервые предложенная Чарльзом Дарвином. Теория подразумевает, что гены приспосабливаются и выживают на протяжении поколений с помощью нескольких факторов. Другими словами, организм с высокими способностями способен выжить в текущей среде и порождает новые организмы в новом поколении, в то время как организм с низкими способностями имеет два шанса выжить в текущей среде и избежать вымирания: вступить в брак с организмом, обладающим высокой выживаемостью, что может привести к появлению в новом поколении особи с высокими способностями, или генетическая мутация, которая может привести к тому, что организм станет сильнее и сможет выжить в текущей среде. Если же организм, полученный в результате одного из двух шансов, не удовлетворяет требованиям среды, он может со временем вымереть. Однако происходит взаимовлияние среды и популяции, а поэтому со временем среда имеет свойство тоже изменяться, и меняются критерии отбора (в нашем случае, они становятся жестче). Для моделирования изменения среды была взята идея об элитизме, при которой лучшая особь популяции переходит в следующее поколение и становится образцом для сравнения показателей других особей. Благодаря этому последующее поколение всегда будет лучше предыдущего или останется таким же по характеристикам. Таким образом, применение идеи теории естественного отбора в генетическом алгоритме улучшит поиск и разнообразие решений обычного генетического алгоритма.

## Общий вид работы алгоритма
1. Указать размер популяции и общее количество итераций алгоритма.
2. Случайным образом сгенерировать начальную популяцию, где каждый ген будет лежать в заданном промежутке, а количество генов будет определяться размерностью функции.
3. Вычислить значение функции для каждой хромосомы в популяции.
4. Вычислить среднее значение полученных результатов.
5. Сравнить значение каждой функции со средним:
   - Если оно меньше, тогда производим операцию мутации над хромосомой, и она переходит в следующее поколение. (1)
   - Если оно больше, тогда у хромосомы есть два шанса на улучшение:
     - Вступить в брак с одной из лучших хромосом. Если полученная хромосома удовлетворяет условию 1, тогда она переходит в последующее поколение.
     - Пройти через мутацию. Если мутированная хромосома удовлетворяет условию 1, тогда она проходит в следующее поколение. Иначе – в поколении генерируется новая, случайная хромосома.

## Мутация
В данном алгоритме используется равномерная мутация, при которой мы выбираем случайный ген в хромосоме особи и заменяем его на новый, случайно сгенерированный.

## Кроссинговер
Алгоритм кроссинговера выполняется по формуле:

𝑛𝑒𝑤 𝑔𝑒𝑛𝑒 = 𝛼[𝑖] ∗ 𝑐ℎ𝑟𝑜𝑚𝑜𝑠𝑜𝑚𝑒[𝑖] + (1 − 𝛼[𝑖]) ∗ 𝑟𝑠𝑐ℎ𝑟𝑜𝑚𝑜𝑠𝑜𝑚𝑒[𝑖]

для каждого гена в хромосоме потомка, где 𝛼 – хромосома, гены которой случайно генерируются в пределах от -𝑔𝑎𝑚𝑚𝑎 до 𝑔𝑎𝑚𝑚𝑎 (являющиеся параметрами алгоритма), 𝑐ℎ𝑟𝑜𝑚𝑜𝑠𝑜𝑚𝑒 – хромосома, вступающая в брак, а 𝑟𝑠𝑐ℎ𝑟𝑜𝑚𝑜𝑠𝑜𝑚𝑒 – одна из пяти лучших хромосом поколения.

# Результаты

Алгоритм достаточно хорошо показал себя на множестве протестированных функций, хоть и не на всех приблизился к минимуму. При этом алгоритм является расширяемым, и можно достичь лучших результатов при изменении его параметров, а также поиска и разработки новых функций мутации и кроссинговера.
