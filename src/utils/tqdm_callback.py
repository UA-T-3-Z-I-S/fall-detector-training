from keras.callbacks import Callback
from tqdm import tqdm

class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.total_batches = self.params['steps']  # total de batches por época

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🌀 Epoch {epoch + 1} comenzando...")
        self.progress_bar = tqdm(total=self.total_batches, desc="📈 Progreso")

    def on_train_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(loss=logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()
