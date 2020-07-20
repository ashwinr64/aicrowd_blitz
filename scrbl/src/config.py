import transformers


MAX_LEN = 160
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "model.bin"
TRAINING_FILE = "../train.csv"
VALID_FILE = "../val.csv"
TOKENIZER = transformers.DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
