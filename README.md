# CASO
Community-Aware Social Community Recommendation

Prerequisites
-------------
* Please refer to requirements.txt

Usage
-----
python run_CASO.py --dataset BlogCatalog --cross_valid=yes --kl_beta=1 --pool_beta=1 --HSIC_lambda=0.01

python run_CASO.py --dataset Flickr --cross_valid=yes --kl_beta=1 --pool_beta=0.4 --HSIC_lambda=0.01

python run_CASO.py --dataset Deezer-HR --cross_valid=yes --kl_beta=0.05 --pool_beta=0.6 --HSIC_lambda=0.1

python run_CASO.py --dataset Deezer-RO --cross_valid=yes --kl_beta=0.5 --pool_beta=0.6 --HSIC_lambda=0.05

python run_CASO.py --dataset=DBLP --cross_valid=yes --kl_beta=0 --pool_beta=1 --HSIC_lambda=0.7  

python run_CASO.py --dataset=Youtube --cross_valid=yes --kl_beta=0.01 --pool_beta=0.9 --HSIC_lambda=0.1 --early_stops=200 

# Dataset
Preprocessed datasets are available at [dataset.zip](https://www.dropbox.com/scl/fo/lyf58kzctrormfajdk8kd/ACUo1wfFDZkYg-UH89Vi5EA?rlkey=sx7sjdk7vxnv03et0qesrsjjf&st=exfoglox&dl=0).

Raw datasets: [BlogCatalog](https://github.com/mengzaiqiao/CAN/tree/master/data), [Flickr](https://github.com/mengzaiqiao/CAN/tree/master/data), [Deezer-HR](https://snap.stanford.edu/data/gemsec-Deezer.html), [Deezer-RO](https://snap.stanford.edu/data/gemsec-Deezer.html), [DBLP](https://snap.stanford.edu/data/com-DBLP.html), [Youtube](https://snap.stanford.edu/data/com-Youtube.html).
