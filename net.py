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

real = ti.f32
number = ti.i32
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
    return 1 / (1 + tm.exp(-x))


@ti.func
def sqrtLU(x: real) -> real:
    return x / (tm.sqrt(1 + ti.abs(x)))


@ti.func
def xavier_U(n_in: number, n_out: number) -> real:
    return ti.sqrt(6 / (n_in + n_out))


@ti.func
def BCE(ideal: real, actual: real):
    lp = ideal * ti.log(actual + 1e-10)
    rp = (1 - ideal) * ti.log(1 - actual + 1e-10)
    return -lp - rp


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
        def __init__(self, num: int, size: tuple = (3, 3), stride: int = 1, padding: int = 1) -> None:
            self.num = num
            self.size = size
            self.stride = stride
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
        self.conv_biases = []
        self.conv_weights = []

        self.dense_outputs = []
        self.dense_outputs_raw = []
        self.dense_biases = []
        self.dense_weights = []

        self._teacher = Teacher.field()
        self._target = scalar()

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

            map_raw_size = ((self.conv_maps[-1].shape[0] + layer.kernel.padding * 2 - layer.kernel.size[0] + 1) // layer.kernel.stride,
                            (self.conv_maps[-1].shape[1] + layer.kernel.padding * 2 - layer.kernel.size[1] + 1) // layer.kernel.stride)
            map_size = (map_raw_size[0] // layer.max_pool[0], 
                        map_raw_size[1] // layer.max_pool[1])

            conv_map_raw = scalar()
            conv_map = scalar()
            conv_weights = scalar()
            conv_biases = scalar()

            map_block = ti.root.dense(ti.k, kernels_num)
            map_block.dense(ti.ij, map_raw_size).place(conv_map_raw)
            map_block.dense(ti.ij, map_size).place(conv_map)

            param_block = ti.root.dense(ti.i, kernels_num)
            param_block.place(conv_biases)

            filter_block = param_block.dense(ti.jk, layer.kernel.size)
            filter_block.dense(ti.l, self.conv_maps[-1].shape[2]).place(conv_weights)

            self.conv_maps_raw.append(conv_map_raw)
            self.conv_maps.append(conv_map)
            self.conv_biases.append(conv_biases)
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
            dense_biases = scalar()

            neuron_block = ti.root.dense(ti.i, neurons_num) 
            neuron_block.place(dense_output_raw, dense_output, dense_biases)
            neuron_block.dense(ti.j, weights_num).place(dense_weights)

            self.dense_outputs_raw.append(dense_output_raw)
            self.dense_outputs.append(dense_output)
            self.dense_biases.append(dense_biases)
            self.dense_weights.append(dense_weights)

        output_size = self.dense_topology[-1].neurons_num
        ti.root.dense(ti.i, output_size).place(self._target)
        ti.root.place(self._teacher)
        ti.root.lazy_grad()
    
    def dump(self, url: str = 'net.model'):
        conv_biases = []
        conv_weights = []
        dense_biases = []
        dense_weights = []

        for l in range(len(self.dense_topology)):
            dense_biases.append(self.dense_biases[l].to_numpy())
            dense_weights.append(self.dense_weights[l].to_numpy())

        for l in range(len(self.conv_topology)):
            conv_biases.append(self.conv_biases[l].to_numpy())
            conv_weights.append(self.conv_weights[l].to_numpy())

        model = {
            'cfg': {
                'input_layer': self.input_layer,
                'conv_topology': self.conv_topology,
                'dense_topology': self.dense_topology
            },
            'dense_weights': dense_weights,
            'dense_biases': dense_biases,
            'conv_weights': conv_weights,
            'conv_biases': conv_biases
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

        dense_biases = model['dense_biases']
        dense_weights = model['dense_weights']
        conv_biases = model['conv_biases']
        conv_weights = model['conv_weights']

        for l in range(len(_net.dense_topology)):
            _net.dense_biases[l].from_numpy(dense_biases[l])
            _net.dense_weights[l].from_numpy(dense_weights[l])

        for l in range(len(_net.conv_topology)):
            _net.conv_biases[l].from_numpy(conv_biases[l])
            _net.conv_weights[l].from_numpy(conv_weights[l])

        return _net
    
    def summary(self):
        table = []

        for l in range(len(self.conv_topology)):
            params_num = (
                self.conv_weights[l].shape[0] 
                * self.conv_weights[l].shape[1] 
                * self.conv_weights[l].shape[2] 
                * self.conv_weights[l].shape[3]
                + self.conv_biases[l].shape[0]
            )
            input_shape = self.conv_maps[l].shape
            output_shape = self.conv_maps[l+1].shape
            table.append([f"Conv__{l+1}", input_shape, output_shape, params_num])

        for l in range(len(self.dense_topology)):
            params_num = (
                self.dense_weights[l].shape[0] 
                * self.dense_weights[l].shape[1]
                + self.dense_biases[l].shape[0]
            )
            input_shape = self.dense_outputs[l].shape
            output_shape = self.dense_outputs[l+1].shape
            table.append([f"Dense_{l+1}", input_shape, output_shape, params_num])

        print(tabulate(table, headers=["Layer", "Input Shape", "Output Shape", "Parameters Number"], tablefmt="pretty"))
        print(f"Total: {self.params_num} parameters\n")

    @ti.kernel
    def _init_params(self):
        cvWs = ti.static(self.conv_weights)
        cvBs = ti.static(self.conv_biases)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        # Xavier Uniform initialization

        for l in ti.static(range(len(cvWs))):
            N = xavier_U(cvWs[l].shape[0], cvWs[l].shape[3])
            for k in range(cvWs[l].shape[0]):
                cvBs[l][k] = 0.
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                                cvWs[l].shape[2],
                                                cvWs[l].shape[3]):
                    cvWs[l][k, kX, kY, kZ] = ti.randn(real) * N

        for l in ti.static(range(len(dsWs))):
            N = xavier_U(dsWs[l].shape[0], dsWs[l].shape[1])
            for n in range(dsWs[l].shape[0]):
                dsBs[l][n] = 0.
                for w in range(dsWs[l].shape[1]):
                    dsWs[l][n, w] = ti.randn(real) * N

    @ti.kernel
    def _clear_grads(self):
        cvMs = ti.static(self.conv_maps)
        cvMs_raw = ti.static(self.conv_maps_raw)
        cvWs = ti.static(self.conv_weights)
        cvBs = ti.static(self.conv_biases)
        dsOs = ti.static(self.dense_outputs)
        dsOs_raw = ti.static(self.dense_outputs_raw)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

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
                cvBs[l].grad[k] = 0.
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
                dsBs[l].grad[n] = 0.
                dsOs[l+1].grad[n] = 0.
                dsOs_raw[l].grad[n] = 0.
                for w in range(dsWs[l].shape[1]):
                    dsWs[l].grad[n, w] = 0.

    @ti.kernel
    def _param_num(self) -> number: 
        cvWs = ti.static(self.conv_weights)
        cvBs = ti.static(self.conv_biases)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        total = 0

        for l in ti.static(range(len(self.conv_topology))):
            total += cvWs[l].shape[0] * cvWs[l].shape[1] * cvWs[l].shape[2] * cvWs[l].shape[3]
            total += cvBs[l].shape[0]

        for l in ti.static(range(len(self.dense_topology))):
            total += dsWs[l].shape[0] * dsWs[l].shape[1]
            total += dsBs[l].shape[0]

        return total

    @ti.kernel
    def _copy_entry(self, entry: ndarray):
        for I in ti.grouped(self.conv_maps[0]):
            self.conv_maps[0][I] = entry[I]

    @ti.kernel
    def _copy_target(self, target: ndarray):
        for n in self._target:
            self._target[n] = target[n]

    @ti.kernel
    def _forward(self):
        cvMs = ti.static(self.conv_maps)
        cvMs_raw = ti.static(self.conv_maps_raw)
        cvWs = ti.static(self.conv_weights)
        cvBs = ti.static(self.conv_biases)
        dsOs = ti.static(self.dense_outputs)
        dsOs_raw = ti.static(self.dense_outputs_raw)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        # loop over conv layers
        for l in ti.static(range(len(cvMs_raw))):
            stride = ti.static(self.conv_topology[l].kernel.stride)
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
                    iX = O.x * stride - padding + kX  # input X
                    iY = O.y * stride - padding + kY  # input Y
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
                    # activation
                    a = sqrtLU(cvMs_raw[l][rX, rY, M.z] + cvBs[l][M.z])
                    # max pooling
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
            for n in dsOs_raw[l]:
                dsOs[l+1][n] = sigmoid(dsBs[l][n] + dsOs_raw[l][n])

    @ti.kernel
    def _calc_penalty(self):
        teacher = ti.static(self._teacher)
        cvBs = ti.static(self.conv_biases)
        cvWs = ti.static(self.conv_weights)
        dsBs = ti.static(self.dense_biases)
        dsWs = ti.static(self.dense_weights)

        for l in ti.static(range(len(cvWs))):
            for B in ti.grouped(cvBs[l]):
                teacher[None].l1 += ti.abs(cvBs[l][B])
                teacher[None].l2 += cvBs[l][B] ** 2
            for K in ti.grouped(cvWs[l]):
                teacher[None].l1 += ti.abs(cvWs[l][K])
                teacher[None].l2 += cvWs[l][K] ** 2

        for l in ti.static(range(len(dsWs))):
            for N in ti.grouped(dsBs[l]):
                teacher[None].l1 += ti.abs(dsBs[l][N])
                teacher[None].l2 += dsBs[l][N] ** 2
            for W in ti.grouped(dsWs[l]):
                teacher[None].l1 += ti.abs(dsWs[l][W])
                teacher[None].l2 += dsWs[l][W] ** 2

    @ti.kernel
    def _compute_loss(self, batch_size: number):
        teacher = ti.static(self._teacher)
        actual = ti.static(self.dense_outputs[-1])
        ideal = ti.static(self._target)

        for n in actual:
            loss = BCE(ideal[n], actual[n])
            penalty = teacher[None].penalty() / self.params_num / actual.shape[0]
            teacher[None].loss += (loss + penalty) / batch_size

    @ti.func
    def _calc_grad_clip(self, grad_threshold: real) -> real:
        cvBs = ti.static(self.conv_biases)
        cvWs = ti.static(self.conv_weights)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        grad_sum = 0.

        for l in ti.static(range(len(cvWs))):
            for k in range(cvWs[l].shape[0]):
                grad_sum += cvBs[l].grad[k] ** 2
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                                cvWs[l].shape[2],
                                                cvWs[l].shape[3]):
                    grad_sum += cvWs[l].grad[k, kX, kY, kZ] ** 2

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                grad_sum += dsBs[l].grad[n] ** 2 
                for w in range(dsWs[l].shape[1]):
                    grad_sum += dsWs[l].grad[n, w] ** 2

        grad_l2 = tm.sqrt(grad_sum)
        grad_clip = grad_threshold / grad_l2

        return tm.clamp(grad_clip, 0., 1.)

    @ti.kernel
    def _advance(self, learn_rate: real, grad_threshold: real):
        cvBs = ti.static(self.conv_biases)
        cvWs = ti.static(self.conv_weights)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        grad_clip = self._calc_grad_clip(grad_threshold)

        for l in ti.static(range(len(cvWs))):
            for k in range(cvWs[l].shape[0]):
                dL_dB = cvBs[l].grad[k] * grad_clip
                cvBs[l][k] -= learn_rate * dL_dB
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1], 
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    dL_dW = cvWs[l].grad[k, kX, kY, kZ] * grad_clip
                    cvWs[l][k, kX, kY, kZ] -= learn_rate * dL_dW

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                dL_dB = dsBs[l].grad[n] * grad_clip
                dsBs[l][n] -= learn_rate * dL_dB
                for w in range(dsWs[l].shape[1]):
                    dL_dW = dsWs[l].grad[n, w] * grad_clip
                    dsWs[l][n, w] -= learn_rate * dL_dW

    def predict(self, entry: np.ndarray) -> np.ndarray:
        self._copy_entry(entry)
        self._forward()
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
            group_label = np.array([0. for _ in range(groups_num)], dtype=np.float32)
            group_label[i] = 1.

            for filename in os.listdir(group_path):
                with Image.open(os.path.join(group_path, filename)) as image:
                    image = image.resize((self.input_layer.size[:2]))

                image = image.convert('RGB')
                image_array = np.array(image, dtype=np.float32) / 255

                imgs.append(image_array)
                lbls.append(group_label)

        return imgs, lbls
    
    def compute_preds_matrix(self, dataset_url: str):

        groups = os.listdir(dataset_url)
        groups_num = len(groups)

        results = np.zeros(shape=(groups_num, groups_num), dtype=np.float32)

        for i in tqdm(range(groups_num), 
                        file=sys.stdout, 
                        unit='classes',
                        desc='Validation progress: ',
                        colour='blue'):
            group_path = os.path.join(dataset_url, groups[i])
            group_samples_num = len(os.listdir(group_path))

            for sample in range(group_samples_num):
                with Image.open(os.path.join(group_path, os.listdir(group_path)[sample])) as image:
                    image = image.resize((self.input_layer.size[:2]))

                image = image.convert('RGB')
                image_array = np.array(image, dtype=np.float32) / 255
            
                results[i] += self.predict(image_array)

            results[i] /= group_samples_num

        return results

    def train(self, train_ds_url: str, 
              val_ds_url: str, 
              epochs: int = 10000,
              batch_size: int = 8,
              history_interval: int = 100,
              learn_rate: float = 0.005, 
              l1_lambda: float = 0.2,
              l2_lambda: float = 0.4,
              grad_threshold: float = 10.5, 
              auto_dump: bool = True,
              dump_interval: int = 2500, 
              dump_url: str = 'net.model'):

        loss_history = np.zeros(shape=(epochs // history_interval, 2))

        self._teacher[None].l1_l = l1_lambda
        self._teacher[None].l2_l = l2_lambda

        train_samples, train_labels = self._collect_dataset(train_ds_url, 'Collecting train dataset: ')
        train_idxs = [np.random.randint(0, len(train_samples), size=batch_size) for _ in range(epochs)]

        val_samples, val_labels = self._collect_dataset(val_ds_url, 'Collecting val dataset: ')
        val_idxs = [np.random.randint(0, len(val_samples), size=batch_size) for _ in range(epochs)]

        train_progress_bar = tqdm(range(epochs),
                                  file=sys.stdout, 
                                  unit='epochs', 
                                  desc='Training progress',
                                  colour='green')
        for epoch in train_progress_bar:
            self._clear_grads()
            self._calc_penalty()

            for i in val_idxs[epoch]:
                self._copy_entry(val_samples[i])
                self._copy_target(val_labels[i])
                self._forward()
                self._compute_loss(batch_size)

            val_loss = self._teacher[None].loss

            self._clear_grads()
            self._calc_penalty()

            for i in train_idxs[epoch]:
                self._copy_entry(train_samples[i])
                self._copy_target(train_labels[i])
                self._forward()
                self._compute_loss(batch_size)

            self._compute_loss.grad(batch_size)
            self._forward.grad()
            self._calc_penalty.grad()

            self._advance(learn_rate, grad_threshold)

            train_loss = self._teacher[None].loss

            loss_history[epoch // history_interval, 0] += train_loss / history_interval
            loss_history[epoch // history_interval, 1] += val_loss / history_interval

            if (epoch + 1) % history_interval == 0:
                tl = loss_history[epoch // history_interval, 0]
                vl = loss_history[epoch // history_interval, 1]
                train_progress_bar.set_postfix_str(f'loss: {tl:0.3f}, val_loss: {vl:0.3f}')

            if auto_dump:
                if (epoch + 1) % dump_interval == 0:
                    curr_lost = loss_history[epoch // history_interval, 0]
                    prev_lost = loss_history[epoch // history_interval - 1, 0]
                    if curr_lost <= prev_lost:
                        self.dump(dump_url)

        return loss_history
