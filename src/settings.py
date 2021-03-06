SEMEVAL_RELATION_LABELS = ['Other', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
                           'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
                           'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
                           'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
                           'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
                           'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
                           'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
                           'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
                           'Content-Container(e1,e2)', 'Content-Container(e2,e1)']

train_path = './drive/MyDrive/data_sets/PERLEX/train.txt'
test_path = './drive/MyDrive/data_sets/PERLEX/test.txt'

train_path_e = './drive/MyDrive/data_sets/PERLEX/train_english.txt'
test_path_e = './drive/MyDrive/data_sets/PERLEX/test_english.txt'
indx2label = dict(enumerate(SEMEVAL_RELATION_LABELS))

label2index = {v: k for k, v in indx2label.items()}

model_name = 'HooshvareLab/bert-fa-zwnj-base'
config = {
    "model_name": model_name,
    "num_labels": len(label2index),
    "max_length": 85,
    'batch_size':16
}
lstm_config = {
    "embedding_dim" : 400,
    "num_hidden_nodes" : 32,
    "num_output_nodes" : len(label2index),
    "num_layers" : 2,
    "bidirection" : True,
    "dropout" : 0.3
}


path_embeddings = '/content/drive/MyDrive/data_sets/PERLEX/embeddings/'
path_fasttext = '/content/drive/MyDrive/data_sets/PERLEX/embeddings/fasttext.bin'
path_word2vec = '/content/drive/MyDrive/data_sets/PERLEX/embeddings/model.txt'

