import os
import sys
import pickle
import numpy as np
import taichi as ti
import taichi.math as tm

from time import time
from tqdm import tqdm
from PIL import Image
from tabulate import tabulate
from matplotlib import pyplot as plt

real = ti.f32
number = ti.i32
template = ti.template()
scalar = lambda: ti.field(dtype=real)
ndarray = ti.types.ndarray(dtype=real)

ti.init(arch=ti.cuda,
        default_fp=real,
        default_ip=number,
        random_seed=int(time()),
        fast_math=False,
        advanced_optimization=True,
        default_gpu_block_dim=1024)


@ti.func
def sigmoid(x: real) -> real:
    return (1 / 2) * (x / (ti.abs(x) + 1) + 1)


@ti.func
def sqd(x: real) -> real:
    return x / tm.sqrt(1 + ti.abs(x))


@ti.func
def BCE(ideal: real, actual: real):
    return -ideal * ti.log(actual + 1e-5) - (1 - ideal) * ti.log(1 - actual + 1e-5)
    # return (ideal - actual) ** 2


@ti.dataclass
class ConvParams:
    bias: real

    @ti.func
    def activate(self, weighted_sum: real) -> real:
        return sqd(weighted_sum + self.bias)


@ti.dataclass
class DenseParams:
    bias: real

    @ti.func
    def activate(self, weighted_sum: real) -> real:
        return sigmoid(weighted_sum + self.bias)


@ti.dataclass
class Teacher:
    l1: real
    l1_l: real
    l2: real
    l2_l: real
    loss: real

    @ti.func
    def penalty(self) -> real:
        return self.l1 * self.l1_l + self.l2 * self.l2_l


@ti.data_oriented
class Net:
    class Kernel:
        def __init__(self, num: int, size: tuple = (3, 3), step: int = 1, padding: int = 1) -> None:
            self.num = num
            self.size = size
            self.step = step
            self.padding = padding

    class Layers:
        class Input:
            def __init__(self, size: tuple = (64, 64, 3)) -> None:
                self.size = size

        class Conv:
            def __init__(self, kernel, max_pool: tuple = (1, 1)) -> None:
                self.kernel = kernel
                self.max_pool = max_pool

        class Dense:
            def __init__(self, neurons_num: int) -> None:
                self.neurons_num = neurons_num

    def __init__(self, input_layer: Layers.Input, 
                 conv_topology: list[Layers.Conv], 
                 dense_topology: list[Layers.Dense]) -> None:

        self.input_layer = input_layer
        self.conv_topology = conv_topology
        self.dense_topology = dense_topology

        self.conv_maps = []
        self.conv_maps_raw = []
        self.conv_params = []
        self.conv_weights = []

        self.dense_outputs = []
        self.dense_outputs_raw = []
        self.dense_params = []
        self.dense_weights = []

        self._teacher = Teacher.field()
        self._loss = scalar()

        self._allocate()
        self._init_params()

        self.params_num = self._param_num()

    def _allocate(self):
        assert self.input_layer != None
        assert self.conv_topology != None
        assert self.dense_topology != None

        conv_map = scalar()
        ti.root.dense(ti.ijk, self.input_layer.size).place(conv_map)
        self.conv_maps.append(conv_map)

        for layer in self.conv_topology:
            kernels_num = layer.kernel.num

            map_raw_size = ((self.conv_maps[-1].shape[0] + layer.kernel.padding * 2 - layer.kernel.size[0] + 1) // layer.kernel.step,
                            (self.conv_maps[-1].shape[1] + layer.kernel.padding * 2 - layer.kernel.size[1] + 1) // layer.kernel.step)
            map_size = (map_raw_size[0] // layer.max_pool[0], 
                        map_raw_size[1] // layer.max_pool[1])

            conv_map_raw = scalar()
            conv_map = scalar()
            conv_weights = scalar()
            conv_params = ConvParams.field()

            map_block = ti.root.dense(ti.k, kernels_num)
            map_block.dense(ti.ij, map_raw_size).place(conv_map_raw)
            map_block.dense(ti.ij, map_size).place(conv_map)

            param_block = ti.root.dense(ti.i, kernels_num)
            param_block.place(conv_params)

            filter_block = param_block.dense(ti.jk, layer.kernel.size)
            filter_block.dense(ti.l, self.conv_maps[-1].shape[2]).place(conv_weights)

            self.conv_maps_raw.append(conv_map_raw)
            self.conv_maps.append(conv_map)
            self.conv_params.append(conv_params)
            self.conv_weights.append(conv_weights)

        dense_output = scalar()
        conv_map_flatten_size = np.prod(self.conv_maps[-1].shape)
        ti.root.dense(ti.i, conv_map_flatten_size).place(dense_output)
        self.dense_outputs.append(dense_output)

        for layer in self.dense_topology:
            neurons_num = layer.neurons_num
            weights_num = self.dense_outputs[-1].shape[0]

            dense_output = scalar()
            dense_output_raw = scalar()
            dense_weights = scalar()
            dense_params = DenseParams.field()

            neuron_block = ti.root.dense(ti.i, neurons_num) 
            neuron_block.place(dense_output_raw, dense_output, dense_params)
            neuron_block.dense(ti.j, weights_num).place(dense_weights)

            self.dense_outputs_raw.append(dense_output_raw)
            self.dense_outputs.append(dense_output)
            self.dense_params.append(dense_params)
            self.dense_weights.append(dense_weights)

        ti.root.place(self._teacher)
        ti.root.place(self._loss)
        ti.root.lazy_grad()
    
    def dump(self, url: str = 'net.model'):
        conv_params = []
        conv_weights = []
        dense_params = []
        dense_weights = []

        for l in range(len(self.dense_topology)):
            dense_params.append(self.dense_params[l].to_numpy())
            dense_weights.append(self.dense_weights[l].to_numpy())

        for l in range(len(self.conv_topology)):
            conv_params.append(self.conv_params[l].to_numpy())
            conv_weights.append(self.conv_weights[l].to_numpy())

        model = {
            'cfg': {
                'input_layer': self.input_layer,
                'conv_topology': self.conv_topology,
                'dense_topology': self.dense_topology
            },
            'dense_weights': dense_weights,
            'dense_params': dense_params,
            'conv_weights': conv_weights,
            'conv_params': conv_params
        }

        with open(url, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(url: str = 'net.model'):
        with open(url, 'rb') as f:
            model = pickle.load(f)
        
        cfg = model['cfg']
        _net = Net(input_layer=cfg['input_layer'], 
                   conv_topology=cfg['conv_topology'],
                   dense_topology=cfg['dense_topology'])

        dense_params = model['dense_params']
        dense_weights = model['dense_weights']
        conv_params = model['conv_params']
        conv_weights = model['conv_weights']

        for l in range(len(_net.dense_topology)):
            _net.dense_params[l].from_numpy(dense_params[l])
            _net.dense_weights[l].from_numpy(dense_weights[l])

        for l in range(len(_net.conv_topology)):
            _net.conv_params[l].from_numpy(conv_params[l])
            _net.conv_weights[l].from_numpy(conv_weights[l])

        return _net
    
    def summary(self):
        table = []

        for l in range(len(self.conv_topology)):
            params_num = (
                self.conv_weights[l].shape[0] * self.conv_weights[l].shape[1] * self.conv_weights[l].shape[2] * self.conv_weights[l].shape[3]
                + self.conv_params[l].shape[0] * 2
            )
            input_shape = self.conv_maps[l].shape
            output_shape = self.conv_maps[l+1].shape
            table.append([f"Conv__{l+1}", input_shape, output_shape, params_num])

        for l in range(len(self.dense_topology)):
            params_num = (
                self.dense_weights[l].shape[0] * self.dense_weights[l].shape[1]
                + self.dense_params[l].shape[0]
            )
            input_shape = self.dense_outputs[l].shape
            output_shape = self.dense_outputs[l+1].shape
            table.append([f"Dense_{l+1}", input_shape, output_shape, params_num])

        print(tabulate(table, headers=["Layer", "Input Shape", "Output Shape", "Parameters Number"], tablefmt="pretty"))
        print(f"Total: {self.params_num} parameters\n")

    @ti.kernel
    def _init_params(self):
        cvWs = ti.static(self.conv_weights)
        cvPs = ti.static(self.conv_params)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        # Xavier Normal initialization
        for l in ti.static(range(len(cvWs))):
            U = 4 * ti.sqrt(2 / (cvWs[l].shape[0] + cvWs[l].shape[3]))
            for k in range(cvWs[l].shape[0]):
                cvPs[l][k].bias = 0.
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    cvWs[l][k, kX, kY, kZ] = ti.randn(real) * U

        # Xavier Uniform initialization
        for l in ti.static(range(len(dsWs))):
            N = ti.sqrt(2 / dsWs[l].shape[1])
            for n in range(dsWs[l].shape[0]):
                dsPs[l][n].bias = 0.
                for w in range(dsWs[l].shape[1]):
                    dsWs[l][n, w] = ti.randn(real) * N

    @ti.kernel
    def _clear_grads(self):
        cvMs = ti.static(self.conv_maps)
        cvMs_raw = ti.static(self.conv_maps_raw)
        cvWs = ti.static(self.conv_weights)
        cvPs = ti.static(self.conv_params)
        dsOs = ti.static(self.dense_outputs)
        dsOs_raw = ti.static(self.dense_outputs_raw)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        self._teacher[None].l1 = 0.
        self._teacher.grad[None].l1 = 0.
        self._teacher[None].l2 = 0.
        self._teacher.grad[None].l2 = 0.

        self._teacher[None].loss = 0.
        self._teacher.grad[None].loss = 1.

        for M in ti.grouped(cvMs[0]):
            cvMs[0].grad[M] = 0.
        for l in ti.static(range(len(cvMs_raw))):
            for k in range(cvWs[l].shape[0]):
                cvPs[l].grad[k].bias = 0.
                for mX, mY in ti.ndrange(cvMs_raw[l].shape[0],
                                         cvMs_raw[l].shape[1]):
                    cvMs[l+1].grad[mX, mY, k] = 0.
                    cvMs_raw[l].grad[mX, mY, k] = 0.
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    cvWs[l].grad[k, kX, kY, kZ] = 0.

        for n in range(dsOs[0].shape[0]):
            dsOs[0].grad[n] = 0.
        for l in ti.static(range(len(self.dense_topology))):
            for n in range(dsWs[l].shape[0]):
                dsOs[l+1].grad[n] = 0.
                dsOs_raw[l].grad[n] = 0.
                dsPs[l].grad[n].bias = 0.
                for w in range(dsWs[l].shape[1]):
                    dsWs[l].grad[n, w] = 0.

    @ti.kernel
    def _param_num(self) -> number: 
        cvWs = ti.static(self.conv_weights)
        cvPs = ti.static(self.conv_params)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        total = 0

        for l in ti.static(range(len(self.conv_topology))):
            total += cvWs[l].shape[0] * cvWs[l].shape[1] * cvWs[l].shape[2] * cvWs[l].shape[3]
            total += cvPs[l].shape[0]

        for l in ti.static(range(len(self.dense_topology))):
            total += dsWs[l].shape[0] * dsWs[l].shape[1]
            total += dsPs[l].shape[0]

        return total

    @ti.kernel
    def _forward(self, entry: ndarray):
        cvMs = ti.static(self.conv_maps)
        cvMs_raw = ti.static(self.conv_maps_raw)
        cvWs = ti.static(self.conv_weights)
        cvPs = ti.static(self.conv_params)
        dsOs = ti.static(self.dense_outputs)
        dsOs_raw = ti.static(self.dense_outputs_raw)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        for I in ti.grouped(cvMs[0]):
            cvMs[0][I] = entry[I]

        # loop over conv layers
        for l in ti.static(range(len(cvMs_raw))):
            step = ti.static(self.conv_topology[l].kernel.step)
            padding = ti.static(self.conv_topology[l].kernel.padding)
            # X compression factor
            mpX = ti.static(self.conv_topology[l].max_pool[0])
            # Y compression factor
            mpY = ti.static(self.conv_topology[l].max_pool[1])

            # loop over feature raw map coords (x, y, z)
            # to calculate the layer's weighted sum
            for O in ti.grouped(cvMs_raw[l]):
                cvMs_raw[l][O] = 0.
                # loop over filter coords (x, y, z)
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    iX = O.x * step - padding + kX  # input X
                    iY = O.y * step - padding + kY  # input Y
                    if (0 <= iX < cvMs[l].shape[0]) and (0 <= iY < cvMs[l].shape[1]):
                        cvMs_raw[l][O] += cvWs[l][O.z, kX, kY, kZ] * cvMs[l][iX, iY, kZ]

            # loop over feature map coords (x, y, z)
            # to activate the weighted sum
            for M in ti.grouped(cvMs[l+1]):
                _max = -tm.inf # init max as -inf
                # loop over max pooling cell
                for cX, cY in ti.ndrange(mpX, mpY):
                    rX = M.x * mpX + cX  # raw X
                    rY = M.y * mpY + cY  # raw Y
                    # activation + max pooling inlined
                    a = cvPs[l][M.z].activate(cvMs_raw[l][rX, rY, M.z])
                    _max = ti.max(_max, a)
                cvMs[l+1][M] = _max

        # flatten the convolutional output
        for I in ti.grouped(cvMs[-1]):
            # flatten index is x * width * depth + y * depth + z
            fI = I.x * cvMs[-1].shape[1] * cvMs[-1].shape[2] + I.y * cvMs[-1].shape[2] + I.z
            dsOs[0][fI] = cvMs[-1][I]

        # loop over fully connected layers
        for l in ti.static(range(len(dsWs))):
            # loop over neurons
            # to calculate the layer's weighted sum
            for n in range(dsWs[l].shape[0]):
                dsOs_raw[l][n] = 0.
                # loop over neuron weights
                for w in range(dsWs[l].shape[1]):
                    dsOs_raw[l][n] += dsWs[l][n, w] * dsOs[l][w]

            # loop over neurons
            # to activate the weighted sum
            if ti.static(l < len(dsWs) - 1):
                for n in dsOs_raw[l]:
                    # sigmoid activation
                    dsOs[l+1][n] = sqd(dsPs[l][n].bias + dsOs_raw[l][n])
            if ti.static(l == len(dsWs) - 1):
                for n in dsOs_raw[l]:
                    dsOs[l+1][n] = sigmoid(dsPs[l][n].bias + dsOs_raw[l][n])

    @ti.kernel
    def _compute_loss(self, ideal: ndarray):
        teacher = ti.static(self._teacher)
        actual = ti.static(self.dense_outputs[-1])

        for n in actual:
            teacher.loss[None] += BCE(ideal[n], actual[n])
            teacher.loss[None] += teacher[None].penalty() / self.params_num
            teacher.loss[None] /= actual.shape[0]

    @ti.kernel
    def _calc_penalty(self):
        teacher = ti.static(self._teacher)
        cvPs = ti.static(self.conv_params)
        cvWs = ti.static(self.conv_weights)
        dsPs = ti.static(self.dense_params)
        dsWs = ti.static(self.dense_weights)

        for l in ti.static(range(len(cvWs))):
            for B in ti.grouped(cvPs[l]):
                teacher[None].l1 += ti.abs(cvPs[l][B].bias)
                teacher[None].l2 += cvPs[l][B].bias ** 2
            for K in ti.grouped(cvWs[l]):
                teacher[None].l1 += ti.abs(cvWs[l][K])
                teacher[None].l2 += cvWs[l][K] ** 2

        for l in ti.static(range(len(dsWs))):
            for N in ti.grouped(dsPs[l]):
                teacher[None].l1 += ti.abs(dsPs[l][N].bias)
                teacher[None].l2 += dsPs[l][N].bias ** 2
            for W in ti.grouped(dsWs[l]):
                teacher[None].l1 += ti.abs(dsWs[l][W])
                teacher[None].l2 += dsWs[l][W] ** 2

    @ti.func
    def _calc_grad_clip(self, grad_threshold: real) -> real:
        cvPs = ti.static(self.conv_params)
        cvWs = ti.static(self.conv_weights)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        grad_sum = 0.

        for l in ti.static(range(len(cvWs))):
            for k in range(cvWs[l].shape[0]):
                grad_sum += cvPs[l].grad[k].bias ** 2
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    grad_sum += cvWs[l].grad[k, kX, kY, kZ] ** 2

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                grad_sum += dsPs[l].grad[n].bias ** 2 
                for w in range(dsWs[l].shape[1]):
                    grad_sum += dsWs[l].grad[n, w] ** 2

        return tm.clamp(grad_threshold / grad_sum, 0., 1.)

    @ti.kernel
    def _advance(self, lr: real, grad_threshold: real):
        cvPs = ti.static(self.conv_params)
        cvWs = ti.static(self.conv_weights)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        grad_clip = self._calc_grad_clip(grad_threshold)

        for l in ti.static(range(len(cvWs))):
            for k in range(cvWs[l].shape[0]):
                dL_dB = cvPs[l].grad[k].bias
                cvPs[l][k].bias -= lr * dL_dB * grad_clip
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1], 
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    dL_dW = cvWs[l].grad[k, kX, kY, kZ]
                    cvWs[l][k, kX, kY, kZ] -= lr * dL_dW * grad_clip

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                dL_dB = dsPs[l].grad[n].bias
                dsPs[l][n].bias -= lr * dL_dB * grad_clip
                for w in range(dsWs[l].shape[1]):
                    dL_dW = dsWs[l].grad[n, w]
                    dsWs[l][n, w] -= lr * dL_dW * grad_clip

    def predict(self, entry: np.ndarray) -> np.ndarray:
        self._forward(entry)
        return self.dense_outputs[-1].to_numpy()

    def _collect_dataset(self, url: str, msg: str):
        imgs = []
        lbls = []

        groups = os.listdir(url)
        groups_num = len(groups)

        for i in tqdm(range(groups_num), 
                      file=sys.stdout, 
                      unit='classes',
                      desc=msg,
                      colour='green'):
            group_path = os.path.join(url, groups[i])

            for filename in os.listdir(group_path):
                image_label = np.array([0. for _ in range(groups_num)], dtype=np.float32)
                image_label[i] = 1.

                with Image.open(os.path.join(group_path, filename)) as image:
                    image = image.resize((self.input_layer.size[0], 
                                            self.input_layer.size[1]))
                    image = image.convert('RGB')
                    image_array = np.array(image, dtype=np.float32) / 255

                imgs.append(image_array)
                lbls.append(image_label)

        return imgs, lbls

    def train(self, train_dataset_url: str, 
              val_dataset_url: str, 
              epochs: int = 10000,
              learn_rate: float = 0.005, 
              l1_lambda: float = 0.2,
              l2_lambda: float = 0.4,
              grad_threshold: float = 10.5, 
              early_stopping: bool = True,
              early_stopping_patience: float = 0.01,
              early_stopping_interval: int = 1000,
              history_interval: int = 100, 
              auto_dump: bool = True,
              dump_interval: int = 2500, 
              dump_url: str = 'net.model'):

        loss_history = np.zeros(shape=(epochs // history_interval, 2))
        self._teacher[None].l1_l = l1_lambda
        self._teacher[None].l2_l = l2_lambda

        train_samples, train_labels = self._collect_dataset(train_dataset_url, 'Collecting train dataset: ')
        val_samples, val_labels = self._collect_dataset(val_dataset_url, 'Collecting val dataset: ')

        train_idxs = np.random.randint(0, len(train_samples), size=epochs)
        val_idxs = np.random.randint(0, len(val_samples), size=epochs)

        progress_bar = tqdm(range(epochs), 
                  file=sys.stdout, 
                  unit='epochs',
                  desc=f'Training progress: ',
                  colour='green')
        for epoch in progress_bar:
            train_idx = train_idxs[epoch]
            val_idx = val_idxs[epoch]

            self._clear_grads()
            self._forward(train_samples[train_idx])
            self._calc_penalty()
            self._compute_loss(train_labels[train_idx])
            self._compute_loss.grad(train_labels[train_idx])
            self._calc_penalty.grad()
            self._forward.grad(train_samples[train_idx])
            self._advance(learn_rate, grad_threshold)

            loss = self._teacher[None].loss / history_interval
            loss_history[epoch // history_interval, 0] += loss

            self._clear_grads()
            self._forward(val_samples[val_idx])
            self._calc_penalty()
            self._compute_loss(val_labels[val_idx])

            loss = self._teacher[None].loss / history_interval
            loss_history[epoch // history_interval, 1] += loss

            if (epoch + 1) % history_interval == 0:
                loss = loss_history[epoch // history_interval, 0]
                val_loss = loss_history[epoch // history_interval, 1]
                progress_bar.set_postfix_str(f'Loss: {loss:0.3f}, Val loss: {val_loss:0.3f}')

            if early_stopping:
                if (epoch + 1) % early_stopping_interval == 0:
                    curr_lost = loss_history[epoch // history_interval, 1]
                    prev_lost = loss_history[epoch // history_interval - 1, 1]
                    if prev_lost - curr_lost <= early_stopping_patience:
                        break

            if auto_dump:
                if (epoch + 1) % dump_interval == 0:
                    curr_lost = loss_history[epoch // history_interval, 1]
                    prev_lost = loss_history[epoch // history_interval - 1, 1]
                    if curr_lost <= prev_lost:
                        self.dump(dump_url)

        return loss_history[:epoch]
    
    def compute_accuracy(self, dataset_url: str):

        groups = os.listdir(dataset_url)
        groups_num = len(groups)

        results = np.zeros(shape=groups_num, dtype=np.float16)

        for i in tqdm(range(groups_num), 
                      file=sys.stdout, 
                      unit=' classes ',
                      desc='Validation progress: ',
                      colour='blue'):
            group_path = os.path.join(dataset_url, groups[i])
            group_samples_num = len(os.listdir(group_path))
            group_res = np.zeros(group_samples_num, dtype=np.float16)

            for sample in range(group_samples_num):
                with Image.open(os.path.join(group_path, os.listdir(group_path)[sample])) as image:
                    image = image.resize((self.input_layer.size[0], self.input_layer.size[1]))
                    image = image.convert('RGB')
                    image_array = np.array(image, dtype=np.float32) / 255

                    self._forward(image_array)
                    group_res[sample] = self.dense_outputs[-1].to_numpy().argmax()
            
            results[i] = np.sum(group_res == i) / group_samples_num

        return results


if __name__ == "__main__":
    # net = Net(
    #     input_layer=Net.Layers.Input(size=(64, 64, 3)),
    #     conv_topology=[
    #         Net.Layers.Conv(Net.Kernel(32, (7, 7), step=2, padding=3)),
    #         Net.Layers.Conv(Net.Kernel(32, (3, 3), step=1, padding=1), max_pool=(2, 2)),
    #         Net.Layers.Conv(Net.Kernel(64, (5, 5), step=1, padding=2), max_pool=(2, 2)),
    #         Net.Layers.Conv(Net.Kernel(64, (3, 3), step=1, padding=1)),
    #         Net.Layers.Conv(Net.Kernel(128, (5, 5), step=2, padding=2)),
    #         Net.Layers.Conv(Net.Kernel(128, (3, 3), step=1, padding=1)),
    #         Net.Layers.Conv(Net.Kernel(256, (3, 3), step=1, padding=1), max_pool=(2, 2)),
    #         Net.Layers.Conv(Net.Kernel(512, (3, 3), step=1, padding=1)),
    #     ],
    #     dense_topology=[
    #         Net.Layers.Dense(neurons_num=1024),
    #         Net.Layers.Dense(neurons_num=256),
    #         Net.Layers.Dense(neurons_num=10)
    #     ]
    # )

    net = Net.load('net.model')

    net.summary()

    history = net.train(train_dataset_url='dataset/training',
                        val_dataset_url='dataset/validation',
                        epochs=100000,
                        learn_rate=.0005,
                        l1_lambda=0.2,
                        l2_lambda=0.4,
                        grad_threshold=5.75,
                        early_stopping=True,
                        early_stopping_patience=0.01,
                        early_stopping_interval=1000,
                        history_interval=1000,
                        auto_dump=True,
                        dump_interval=25000,
                        dump_url='net.model')

    print(net.compute_accuracy('dataset/validation'))

    plt.plot(history)
    plt.show()
