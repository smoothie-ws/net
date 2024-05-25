import os
import sys
import pickle
import numpy as np
import taichi as ti
import taichi.math as tm

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

real = ti.f32
number = ti.i32
template = ti.template()
scalar = lambda: ti.field(dtype=real)
ndarray = ti.types.ndarray(dtype=real)

ti.init(arch=ti.vulkan,
        default_fp=real,
        default_ip=number)


@ti.func
def sigmoid(x: real) -> real:
    return 1 / (1 + tm.exp(-x))
    

@ti.func
def logshift(x: real, base: real) -> real:
    res = tm.log(1 + ti.abs(x)) / tm.log(base)
    if x < 0:
        res *= -1
    return res


@ti.func
def BCE(ideal: real, actual: real) -> real:
    return -(ideal * ti.log(actual) + (1 - ideal) * ti.log(1 - actual))


@ti.dataclass
class ConvNeuron:
    alpha: real
    base: real
    bias: real

    @ti.func
    def activate(self, weighted_sum: real) -> real:
        a = logshift(weighted_sum + self.bias, self.base) * self.alpha
        return a


@ti.dataclass
class DenseNeuron:
    bias: real

    @ti.func
    def activate(self, weighted_sum: real) -> real:
        a = sigmoid(weighted_sum + self.bias)
        return a


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
        """Initialize a CNN instance.

        Args:
            input_layer: `tuple` - 3-dimensional size of the input layer.
            conv_topology: `list` - Topology of the convolutional layers. 
                                    Each layer must be a list of a 3-dimensional 
                                    feature map size and a 2-dimensional filter size.
            dense_topology: `list` - Topology of the fully connected layers.
                                    Each layer must be a number of neurons it contains.

        Example::

            >>> net = ConvNet(
            >>>     input_size=(64, 64, 3),
            >>>     conv_topology=[[(30, 30, 6), (3, 3)],
            >>>                  [(14, 14, 3), (3, 3)]],
            >>>     dense_topology=[64, 2]
            >>> )
        """

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

        self._loss = scalar()
        self._loss_penalty_l1 = scalar()
        self._loss_penalty_l2 = scalar()

        self._allocate()
        self._init_params()

        self.param_num = self._param_num()

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
            conv_params = ConvNeuron.field()

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
            dense_params = DenseNeuron.field()

            neuron_block = ti.root.dense(ti.i, neurons_num) 
            neuron_block.place(dense_output_raw, dense_output, dense_params)
            neuron_block.dense(ti.j, weights_num).place(dense_weights)

            self.dense_outputs_raw.append(dense_output_raw)
            self.dense_outputs.append(dense_output)
            self.dense_params.append(dense_params)
            self.dense_weights.append(dense_weights)

        ti.root.place(self._loss_penalty_l1, self._loss_penalty_l2, self._loss)
        ti.root.lazy_grad()
    
    def dump_params(self, url: str = 'params.net'):
        """Save a CNN instance's parameters.

        Args:
            url: `str` - path to the file to save the parameters.

        Example::

            >>> net.dump_params('params.net')
        """

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

        params = {
            'dense_weights': dense_weights,
            'dense_params': dense_params,
            'conv_weights': conv_weights,
            'conv_params': conv_params
        }

        with open(url, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, url: str = 'params.net'):
        """Load a CNN instance's parameters.
        
        Args:
            url: `str` - path to the file to load the parameters.

        Example::

            >>> net.load_params('params.net')
        """

        with open(url, 'rb') as f:
            params = pickle.load(f)

        dense_params = params['dense_params']
        dense_weights = params['dense_weights']
        conv_params = params['conv_params']
        conv_weights = params['conv_weights']

        for l in range(len(self.dense_topology)):
            self.dense_params[l].from_numpy(dense_params[l])
            self.dense_weights[l].from_numpy(dense_weights[l])

        for l in range(len(self.conv_topology)):
            self.conv_params[l].from_numpy(conv_params[l])
            self.conv_weights[l].from_numpy(conv_weights[l])

    @ti.kernel
    def _init_params(self):
        cvWs = ti.static(self.conv_weights)
        cvPs = ti.static(self.conv_params)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        # He initialization for the conv layers
        for l in ti.static(range(len(cvWs))):
            u = ti.sqrt(2 / (cvWs[l].shape[3]))
            for k in range(cvWs[l].shape[0]):
                cvPs[l][k].alpha = 1.
                cvPs[l][k].base = tm.e
                cvPs[l][k].bias = 0.
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    cvWs[l][k, kX, kY, kZ] = ti.randn(real) * u

        # Xavier initialization for the dense layers
        for l in ti.static(range(len(dsWs))):
            u = ti.sqrt(6 / (dsWs[l].shape[0] + dsWs[l].shape[1]))
            for n in range(dsWs[l].shape[0]):
                dsPs[l][n].bias = 0.
                for w in range(dsWs[l].shape[1]):
                    dsWs[l][n, w] = ti.randn(real) * u

    @ti.kernel
    def _param_num(self) -> number:       
        """Return a total number of a CNN instance's parameters
        
        Example::

            >>> net = ConvNet(
            >>>     input_size=(64, 64, 3),
            >>>     conv_topology=[[(30, 30, 32), (3, 3)],
            >>>                  [(14, 14, 64), (5, 5)]],
            >>>     dense_topology=[512, 32, 4]
            >>> )
            >>> print(net.param_num)
            >>> 6491748
            """
        cvWs = ti.static(self.conv_weights)
        cvPs = ti.static(self.conv_params)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        total = 0

        for l in ti.static(range(len(self.conv_topology))):
            total += cvWs[l].shape[0] * cvWs[l].shape[1] * cvWs[l].shape[2] * cvWs[l].shape[3]
            total += cvPs[l].shape[0] * 2

        for l in ti.static(range(len(self.dense_topology))):
            total += dsWs[l].shape[0] * dsWs[l].shape[1]
            total += dsPs[l].shape[0]

        return total

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

        self._loss[None] = 0.
        self._loss.grad[None] = 1.
        self._loss_penalty_l1[None] = 0.
        self._loss_penalty_l2[None] = 0.

        for O in ti.grouped(cvMs[0]):
            cvMs[0].grad[O] = 0.
        for l in ti.static(range(len(cvMs_raw))):
            for k in range(cvWs[l].shape[0]):
                cvPs[l].grad[k].alpha = 0.
                cvPs[l].grad[k].bias = 0.
                cvPs[l].grad[k].base = 0.
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1], 
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    cvWs[l].grad[k, kX, kY, kZ] = 0.
            for O in ti.grouped(cvMs_raw[l]):
                cvMs_raw[l].grad[O] = 0.
            for O in ti.grouped(cvMs[l+1]):
                cvMs[l+1].grad[O] = 0.

        for n in dsOs[0]:
            dsOs[0].grad[n] = 0.
        for l in ti.static(range(len(dsOs_raw))):
            for n in range(dsWs[l].shape[0]):
                dsOs[l+1].grad[n] = 0.
                dsOs_raw[l].grad[n] = 0.
                dsPs[l].grad[n].bias = 0.
                for w in range(dsWs[l].shape[1]):
                    dsWs[l].grad[n, w] = 0.

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
                    # ELU activation + max pooling inlined
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
            for n in dsOs_raw[l]:
                # sigmoid activation
                dsOs[l+1][n] = dsPs[l][n].activate(dsOs_raw[l][n])

    @ti.kernel
    def _compute_loss(self, ideal: ndarray, l1_lambda: real, l2_lambda: real):
        loss = ti.static(self._loss)
        actual = ti.static(self.dense_outputs[-1])
        cvPs = ti.static(self.conv_params)
        cvWs = ti.static(self.conv_weights)
        dsPs = ti.static(self.dense_params)
        dsWs = ti.static(self.dense_weights)
        penalty_l1 = ti.static(self._loss_penalty_l1)
        penalty_l2 = ti.static(self._loss_penalty_l2)

        for l in ti.static(range(len(cvWs))):
            for k in range(cvWs[l].shape[0]):
                penalty_l1[None] += ti.abs(cvPs[l][k].alpha)
                penalty_l1[None] += ti.abs(cvPs[l][k].bias)
                penalty_l1[None] += ti.abs(cvPs[l][k].base)
                penalty_l2[None] += cvPs[l].grad[k].alpha ** 2
                penalty_l2[None] += cvPs[l].grad[k].bias ** 2
                penalty_l2[None] += cvPs[l].grad[k].base ** 2
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    penalty_l1[None] += ti.abs(cvWs[l].grad[k, kX, kY, kZ]) ** 2
                    penalty_l2[None] += cvWs[l].grad[k, kX, kY, kZ] ** 2

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                penalty_l1[None] += ti.abs(dsPs[l][n].bias)
                penalty_l2[None] += dsPs[l][n].bias ** 2
                for w in range(dsWs[l].shape[1]):
                    penalty_l1[None] += ti.abs(dsWs[l][n, w])
                    penalty_l2[None] += dsWs[l][n, w] ** 2

        for n in actual:
            loss[None] += BCE(ideal[n],actual[n])
            loss[None] += penalty_l1[None] * l1_lambda / self.param_num
            loss[None] += penalty_l2[None] * l2_lambda / self.param_num
            loss[None] *= 1 / actual.shape[0]

    @ti.kernel
    def _advance(self, lr: real, grad_threshold: real):
        cvPs = ti.static(self.conv_params)
        cvWs = ti.static(self.conv_weights)
        dsWs = ti.static(self.dense_weights)
        dsPs = ti.static(self.dense_params)

        grad_sum = 0.

        for l in ti.static(range(len(cvWs))):
            for k in range(cvWs[l].shape[0]):
                grad_sum += cvPs[l].grad[k].alpha ** 2
                grad_sum += cvPs[l].grad[k].bias ** 2
                grad_sum += cvPs[l].grad[k].base ** 2
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1],
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    grad_sum += cvWs[l].grad[k, kX, kY, kZ] ** 2

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                grad_sum += dsPs[l].grad[n].bias ** 2 
                for w in range(dsWs[l].shape[1]):
                    grad_sum += dsWs[l].grad[n, w] ** 2
        
        grad_clip = tm.clamp(grad_threshold / grad_sum, 0., 1.)

        for l in ti.static(range(len(cvWs))):
            for k in range(cvWs[l].shape[0]):
                dL_dB = cvPs[l].grad[k].bias * grad_clip
                dL_db = cvPs[l].grad[k].base * grad_clip
                dL_dA = cvPs[l].grad[k].alpha * grad_clip
                cvPs[l][k].base -= lr * dL_db
                cvPs[l][k].bias -= lr * dL_dB
                cvPs[l][k].alpha -= lr * dL_dA
                for kX, kY, kZ in ti.ndrange(cvWs[l].shape[1], 
                                             cvWs[l].shape[2],
                                             cvWs[l].shape[3]):
                    dL_dW = cvWs[l].grad[k, kX, kY, kZ] * grad_clip
                    cvWs[l][k, kX, kY, kZ] -= lr * dL_dW

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                dL_dB = dsPs[l].grad[n].bias * grad_clip
                dsPs[l][n].bias -= lr * dL_dB
                for w in range(dsWs[l].shape[1]):
                    dL_dW = dsWs[l].grad[n, w] * grad_clip
                    dsWs[l][n, w] -= lr * dL_dW

    def predict(self, entry: np.ndarray) -> np.ndarray:
        """Pass the entry through the CNN instance and 
        return the very last fully connected layer output.

        Args:
            entry: `ndarray`
        Returns:
            output: `np.ndarray`

        Example::

            >>> entry = np.array(...)
            >>> res = net.predict(entry)
            >>> print(res)
            >>> [0.01729, 0.97124, ... ]
        """

        self._forward(entry)
        return self.dense_outputs[-1].to_numpy()

    def _collect_dataset(self, url: str):
        imgs = []
        lbls = []

        groups = os.listdir(url)
        groups_num = len(groups)

        for i in tqdm(range(groups_num), 
                      file=sys.stdout, 
                      unit='classes',
                      desc='Collecting dataset: ',
                      colour='green'):
            group_path = os.path.join(url, groups[i])

            for filename in os.listdir(group_path):
                image_label = np.array([0. for _ in range(groups_num)], dtype=np.float32)
                image_label[i] = 1.

                try:
                    with Image.open(os.path.join(group_path, filename)) as image:
                        image = image.resize((self.input_layer.size[0], self.input_layer.size[1]))
                        image = image.convert('RGB')
                        image_array = np.array(image, dtype=np.float32) / 255

                    imgs.append(image_array)
                    lbls.append(image_label)
                except:
                    pass

        return imgs, lbls

    def train(self, dataset_url: str, 
              epochs: int = 1000, 
              learn_rate: float = 0.005, 
              l1_lambda: float = 0.1,
              l2_lambda: float = 0.2,
              grad_threshold: float = 0.5, 
              history_interval: int = 10, 
              dump_interval: int = 100, 
              dump_url: str = 'params.net'):
        """Train a CNN instance.

        Args:
            training_url: `str` - path to the dataset.
            epochs: `int` - number of epochs.
            batch_size: `int` - size of a batch.
            learn_rate: `float` - learn rate.
        Returns:
            history: `np.ndarray` - loss history

        Example::

            >>> history = net.train(
            >>>     samples=samples, 
            >>>     targets=targets,
            >>>     epochs=150,
            >>>     batch_size=64,
            >>>     learn_rate=0.01
            >>> )
            >>> plt.plot(history)
            >>> plt.show()
        """

        loss_history = np.zeros(epochs // history_interval)

        imgs, lbls = self._collect_dataset(dataset_url)
        samples_num = len(imgs)

        idxs = np.arange(samples_num)
        np.random.shuffle(idxs)

        for epoch in tqdm(range(epochs), 
                          file=sys.stdout, 
                          unit='epochs',
                          desc=f'Training progress: ',
                          colour='green'):

            self._clear_grads()
            idx = epoch % samples_num

            self._forward(imgs[idxs[idx]])
            self._compute_loss(lbls[idxs[idx]], l1_lambda, l2_lambda)
            self._compute_loss.grad(lbls[idxs[idx]], l1_lambda, l2_lambda)
            self._forward.grad(imgs[idxs[idx]])

            # params correction
            self._advance(learn_rate, grad_threshold)

            if (epoch + 1) % dump_interval == 0:
                self.dump_params(dump_url)

            loss_history[epoch // history_interval] += self._loss[None] / history_interval

        return loss_history
    
    def compute_accuracy(self, dataset_url: str):
        """Compute the approximate accuracy of the model.

        Args:
            training_url: `str` - path to the dataset.
        Returns:
            accuracy: `float` - approximate accuracy

        Example::

            >>> accuracy = net.compute_accuracy('dataset')
            >>> print(accuracy)
            >>> 0.87921
        """

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
                try:
                    with Image.open(os.path.join(group_path, os.listdir(group_path)[sample])) as image:
                        image = image.resize((self.input_layer.size[0], self.input_layer.size[1]))
                        image = image.convert('RGB')
                        image_array = np.array(image, dtype=np.float32) / 255

                        self.conv_maps[0].from_numpy(image_array)
                        self._forward()
                        group_res[sample] = self.dense_outputs[-1].to_numpy().argmax()
                except:
                    pass

            results[i] = np.sum(group_res == i) / group_samples_num

        return results


if __name__ == "__main__":
    net = Net(
        input_layer=Net.Layers.Input(size=(64, 64, 3)),
        conv_topology=[
            Net.Layers.Conv(Net.Kernel(32, (3, 3), step=2, padding=1), max_pool=(2, 2)),
            Net.Layers.Conv(Net.Kernel(64, (5, 5), step=1, padding=2), max_pool=(2, 2)),
            Net.Layers.Conv(Net.Kernel(128, (5, 5), step=2, padding=2), max_pool=(2, 2)),
            Net.Layers.Conv(Net.Kernel(256, (3, 3), step=2, padding=1), max_pool=(1, 1))
        ],
        dense_topology=[
            Net.Layers.Dense(neurons_num=10)
        ]
    )

    print(net.param_num)  # 555306

    imgs = [Image.open('dataset/training/dew/2208.jpg').resize((64, 64)).convert('RGB'),
            Image.open('dataset/training/fog/4075.jpg').resize((64, 64)).convert('RGB'),
            Image.open('dataset/training/frost/4930.jpg').resize((64, 64)).convert('RGB'),
            Image.open('dataset/training/hail/0000.jpg').resize((64, 64)).convert('RGB')]

    for img in imgs:
        arr = np.array(img, dtype=np.float32) / 255
        print(net.predict(arr))

    history = net.train(dataset_url='dataset/training',
                        epochs=100000,
                        learn_rate=0.005,
                        l1_lambda=0.2,
                        l2_lambda=0.4,
                        grad_threshold=17.5,
                        history_interval=100,
                        dump_interval=1000,
                        dump_url='params.net')

    for img in imgs:
        arr = np.array(img, dtype=np.float32) / 255
        print(net.predict(arr))

    plt.plot(history)
    plt.show()
