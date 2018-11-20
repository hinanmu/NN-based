# NN-Based
## Dataset
[http://mulan.sourceforge.net/datasets-mlc.html][1]

### yeast
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|

## Evaluation
|evaluation criterion |NN-Based |
| ------ | ------ | 
| hamming loss| 0.247546346782988 |

## Requrements
- Python 3.6
- numpy 1.13.3
- tensorflow 1.10.0
- scikit-learn 0.19.1

## Parameter
- hidden_unit:0.8 * feature number
- Regularization alpha:0.1
- learning rate:0.05
- trainning step:500 * batch number

## Reference
[J. Nam, J. Kim, E. Loza Menc ́ıa, I. Gurevych, andJ. F ̈urnkranz. Large-scale multi-label text classification —revisiting neural networks. InECML PKDD 2014][2]


  [1]: http://mulan.sourceforge.net/datasets-mlc.html
  [2]: https://link.springer.com/chapter/10.1007/978-3-662-44851-9_28





