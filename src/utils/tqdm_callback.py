from keras.callbacks import Callback
from tqdm import tqdm

class TQDMProgressBar(Callback):
    def __init__(self, update_every=10):
        super().__init__()
        self.update_every = update_every
        self.last_logged_batch = 0

    def on_train_begin(self, logs=None):
        self.total_batches = self.params['steps']  # total de batches por época

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🌀 Epoch {epoch + 1} comenzando...")
        self.progress_bar = tqdm(
            total=self.total_batches,
            desc="📈 Progreso",
            dynamic_ncols=True
        )
        self.last_logged_batch = 0

    def on_train_batch_end(self, batch, logs=None):
        # Actualiza la barra cada N batches
        if batch - self.last_logged_batch >= self.update_every or batch == self.total_batches - 1:
            postfix_data = {}
            if logs is not None:
                if 'loss' in logs:
                    postfix_data['loss'] = f"{logs['loss']:.4f}"
                if 'accuracy' in logs:
                    postfix_data['acc'] = f"{logs['accuracy']:.4f}"
            self.progress_bar.update(batch - self.last_logged_batch)
            self.progress_bar.set_postfix(postfix_data)
            self.last_logged_batch = batch

    def on_epoch_end(self, epoch, logs=None):
        # Asegura cerrar la barra correctamente
        remaining = self.total_batches - self.last_logged_batch
        if remaining > 0:
            self.progress_bar.update(remaining)
        self.progress_bar.close()
