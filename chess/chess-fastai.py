!rm -rf data
!mkdir data
!wget https://s3.eu-central-1.wasabisys.com/aicrowd-practice-challenges/public/chess/v0.1/test.csv
!wget https://s3.eu-central-1.wasabisys.com/aicrowd-practice-challenges/public/chess/v0.1/train.csv
!mv train.csv data/train.csv
!mv test.csv data/test.csv

from fastai2.tabular.all import *


df = pd.read_csv('data/train.csv')
df.head()

dls = TabularDataLoaders.from_csv('data/train.csv', y_names="6",
    cat_names = ['0', '1', '2', '3', '4', '5'],
    cont_names = [],
    procs = [Categorify, Normalize],
    y_block = CategoryBlock, bs=32)

learn = tabular_learner(dls, metrics=[accuracy, F1Score(average='macro')], layers=[1000, 1000, 500])
learn.model

learn.fit_one_cycle(5, max_lr=1e-2)

dl = learn.dls.test_dl(test_df)
op = learn.get_preds(dl=dl)
pred = op[0].numpy()
final_pred = [i.argmax() for i in pred]

submission = pd.DataFrame(final_pred)
# submission = submission.round(3)
submission.to_csv('submission.csv',header=['depth'],index=False)
submission.to_csv('submission.csv',header=['depth'],index=False)