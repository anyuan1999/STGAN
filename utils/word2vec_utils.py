from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def on_epoch_end(self, model):
        model.save(f'word2vec_{self.dataset_name}.model')


class EpochLogger(CallbackAny2Vec):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def on_epoch_begin(self, model):
        print(f"{self.dataset_name} - Epoch start")

    def on_epoch_end(self, model):
        print(f"{self.dataset_name} - Epoch end")


def train_word2vec(phrases, dataset_name, vector_size=30, window=5, min_count=1, workers=8, epochs=300):
    """Train a Word2Vec model with the given parameters."""
    saver = EpochSaver(dataset_name)
    logger = EpochLogger(dataset_name)

    model = Word2Vec(sentences=phrases, vector_size=vector_size, window=window,
                     min_count=min_count, workers=workers, epochs=epochs,
                     callbacks=[saver, logger])
    return model

