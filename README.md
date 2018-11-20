# NN-Based
## Dataset
[http://mulan.sourceforge.net/datasets-mlc.html][1]

### yeast
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|
| bookmarks | text | 87856	 |2150|0|208|2.028|0.010|18716|
| delicious|text (web) | 16105	|500|0|983|19.020|0.019|15806|

## Evaluation
|evaluation criterion |NN-Based |
| ------ | ------ | 
| hamming loss|   |

## Requrements
- Python 3.6
- numpy 1.13.3
- tensorflow 1.10.0
- scikit-learn 0.19.1

## Parameter
- hidden_unit:20 to 4000
- Regularization alpha:0.1
- learning rate:0.05
- trainning epoch:100

## Reference
[J. Nam, J. Kim, E. Loza Menc ́ıa, I. Gurevych, andJ. F ̈urnkranz. Large-scale multi-label text classification —revisiting neural networks. InECML PKDD 2014][2]


  [1]: http://mulan.sourceforge.net/datasets-mlc.html
  [2]: https://link.springer.com/chapter/10.1007/978-3-662-44851-9_28





