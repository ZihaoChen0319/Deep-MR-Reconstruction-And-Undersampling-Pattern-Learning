import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

from Runtime.Losses import DiceCELoss
from Runtime.Metrics import Dice


class TrainEngine:
    def __init__(self, model, train_ds, val_ds, n_epochs, n_class, loss_weights, val_freq=5):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.n_epochs = n_epochs
        self.n_class = n_class
        self.loss_weights = loss_weights
        self.val_freq = val_freq
        self.optimizer = Adam(learning_rate=1e-3)
        self.loss_fn_dice = DiceCELoss(y_one_hot=True, reduce_batch=True, include_background=False)
        self.loss_fn_mse = MeanSquaredError()
        self.metric_fns = [tf.keras.metrics.MeanSquaredError()] + \
                          [Dice(n_class=n_class, index=i) for i in range(1, n_class)]

    def reset_metrics(self):
        for i in range(len(self.metric_fns)):
            self.metric_fns[i].reset_state()

    # @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            recon, seg = self.model(x, training=True)
            loss = self.loss_fn_mse(y_true=x, y_pred=recon) * self.loss_weights[0] \
                   + self.loss_fn_dice(y_true=y, y_pred=seg) * self.loss_weights[1]
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metric_fns[0].update_state(y_true=x, y_pred=recon)
        for i in range(1, self.n_class):
            self.metric_fns[i].update_state(y_true=y, y_pred=seg)
        return loss

    # @tf.function
    def eval_step(self, x, y):
        recon, seg = self.model(x, training=False)
        self.metric_fns[0].update_state(y_true=x, y_pred=recon)
        for i in range(1, self.n_class):
            self.metric_fns[i].update_state(y_true=y, y_pred=seg)

    def train(self, if_task_adaptive=True):
        self.set_recon_layer(if_trainable=True)
        self.set_seg_layer(if_trainable=False)
        seg_weight = self.loss_weights[1]
        self.loss_weights[1] = 0

        for ep in range(1, self.n_epochs + 1):
            self.train_ds.shuffle(buffer_size=100)
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                loss = self.train_step(x_batch_train, y_batch_train)

            train_metrics = [self.metric_fns[i].result().numpy() for i in range(len(self.metric_fns))]
            print('Epoch %d/%d, loss %.4f, mse %.6f, dice' %
                  (ep, self.n_epochs, loss, train_metrics[0]), train_metrics[1:])
            self.reset_metrics()

            if ep % self.val_freq == 0:
                self.evaluate()
                self.reset_metrics()

            if ep == (self.n_epochs // 2):
                if if_task_adaptive:
                    self.set_recon_layer(if_trainable=True)
                    self.set_seg_layer(if_trainable=True)
                    self.loss_weights[1] = seg_weight
                else:
                    self.set_recon_layer(if_trainable=False)
                    self.set_seg_layer(if_trainable=True)
                    self.loss_weights[1] = seg_weight

        self.set_seg_layer(if_trainable=True)
        self.set_seg_layer(if_trainable=True)

    def evaluate(self):
        for step, (x_batch_train, y_batch_train) in enumerate(self.val_ds):
            self.eval_step(x_batch_train, y_batch_train)

        test_metrics = [self.metric_fns[i].result().numpy() for i in range(len(self.metric_fns))]
        print('Test mse %.6f, dice' % test_metrics[0], test_metrics[1:])

        self.reset_metrics()

    def set_recon_layer(self, if_trainable=True):
        for layer in self.model.layers:
            layer.trainable = if_trainable
            if layer.name == 'recon_final':
                break

    def set_seg_layer(self, if_trainable=True):
        flag = False
        for layer in self.model.layers:
            if flag:
                layer.trainable = if_trainable
            if layer.name == 'recon_final':
               flag = True

