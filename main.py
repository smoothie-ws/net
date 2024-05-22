import os
import sys
import pickle
import numpy as np
import taichi as ti
import taichi.math as tm

from tqdm import tqdm
from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt

real = ti.f32
number = ti.i32
template = ti.template()
scalar = lambda: ti.field(dtype=real)
ndarray = ti.types.ndarray(dtype=real)

ti.init(arch=ti.cuda,
        default_fp=real,
        default_ip=number)


@ti.func
def sigmoid(x: real) -> real:
    return 1 / (1 + tm.exp(-x))


@ti.func
def ELU(x: real) -> real:
    res = x
    if x < 0:
        res = tm.exp(x) - 1
    return res


@ti.func
def BCE(ideal: real, actual: real) -> real:
    L = ideal * ti.log(actual) + (1 - ideal) * ti.log(1 - actual)
    return -L

@ti.func
def MSE(ideal: real, actual: real) -> real:
    L = (ideal - actual) ** 2
    return L

@ti.dataclass
class BNparams:
    g: real  # gamma
    b: real  # beta

    @ti.func
    def norm(self, x: real, mx: real, dx: real) -> real:
        x_norm = (x - mx) / ti.sqrt(dx + 1e-5)
        return self.g * x_norm + self.b
    

@ti.dataclass
class BNlayer:
    mx: real
    dx: real
    

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
        self.conv_bn_params = []
        self.conv_kernels = []

        self.dense_outputs = []
        self.dense_outputs_raw = []
        self.dense_biases = []
        self.dense_weights = []

        self._loss = scalar()
        self._conv_bn = []

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
            conv_kernels= scalar()
            conv_bn = BNlayer.field()
            conv_bn_params = BNparams.field()

            map_block = ti.root.dense(ti.k, kernels_num)
            map_block.dense(ti.ij, map_raw_size).place(conv_map_raw)
            map_block.dense(ti.ij, map_size).place(conv_map)

            param_block = ti.root.dense(ti.i, kernels_num)
            param_block.place(conv_bn, conv_bn_params)

            filter_block = param_block.dense(ti.jk, layer.kernel.size)
            filter_block.dense(ti.l, self.conv_maps[-1].shape[2]).place(conv_kernels)

            self.conv_maps_raw.append(conv_map_raw)
            self.conv_maps.append(conv_map)
            self._conv_bn.append(conv_bn)
            self.conv_bn_params.append(conv_bn_params)
            
            self.conv_kernels.append(conv_kernels)

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

        # ti.root.dense(ti.i, self.dense_topology[-1].neurons_num).place(self._loss)
        ti.root.place(self._loss)
        
        ti.root.lazy_grad()
    
    def dump_params(self, url: str = 'params.net'):
        """Save a CNN instance's parameters.

        Args:
            url: `str` - path to the file to save the parameters.

        Example::

            >>> net.dump_params('params.net')
        """

        conv_bn_params = []
        conv_kernels = []
        dense_biases = []
        dense_weights = []

        for l in range(len(self.dense_topology)):
            dense_biases.append(self.dense_biases[l].to_numpy())
            dense_weights.append(self.dense_weights[l].to_numpy())

        for l in range(len(self.conv_topology)):
            conv_bn_params.append(self.conv_bn_params[l].to_numpy())
            conv_kernels.append(self.conv_kernels[l].to_numpy())

        params = {
            'dense_weights': dense_weights,
            'dense_biases': dense_biases,
            'conv_kernels': conv_kernels,
            'conv_bn_params': conv_bn_params
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

        dense_biases = params['dense_biases']
        dense_weights = params['dense_weights']
        conv_bn_params = params['conv_bn_params']
        conv_kernels = params['conv_kernels']

        for l in range(len(self.dense_topology)):
            self.dense_biases[l].from_numpy(dense_biases[l])
            self.dense_weights[l].from_numpy(dense_weights[l])

        for l in range(len(self.conv_topology)):
            self.conv_bn_params[l].from_numpy(conv_bn_params[l])
            self.conv_kernels[l].from_numpy(conv_kernels[l])

    @ti.kernel
    def _init_params(self):
        cvKs = ti.static(self.conv_kernels)
        cvBN_params = ti.static(self.conv_bn_params)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        # He initialization for the conv layers
        for l in ti.static(range(len(cvKs))):
            u = ti.sqrt(2 / (cvKs[l].shape[3]))
            for k in range(cvKs[l].shape[0]):
                cvBN_params[l][k].g = 1.
                cvBN_params[l][k].b = 0.
                for kX, kY, kZ in ti.ndrange(cvKs[l].shape[1],
                                             cvKs[l].shape[2],
                                             cvKs[l].shape[3]):
                    cvKs[l][k, kX, kY, kZ] = ti.randn(real) * u

        # Xavier initialization for the dense layers
        for l in ti.static(range(len(dsWs))):
            u = ti.sqrt(6 / (dsWs[l].shape[0] + dsWs[l].shape[1]))
            for n in range(dsWs[l].shape[0]):
                dsBs[l][n] = 0.
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
        cvKs = ti.static(self.conv_kernels)
        cvBN_params = ti.static(self.conv_bn_params)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        total = 0

        for l in ti.static(range(len(self.conv_topology))):
            total += cvKs[l].shape[0] * cvKs[l].shape[1] * cvKs[l].shape[2] * cvKs[l].shape[3]
            total += cvBN_params[l].shape[0] * 2

        for l in ti.static(range(len(self.dense_topology))):
            total += dsWs[l].shape[0] * dsWs[l].shape[1]
            total += dsBs[l].shape[0]

        return total
    
    @ti.kernel
    def _sum_params_l2(self) -> real:
        cvKs = ti.static(self.conv_kernels)
        cvBN_params = ti.static(self.conv_bn_params)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        _sum = 0.
        for l in ti.static(range(len(cvKs))):
            for k in range(cvKs[l].shape[0]):
                _sum += cvBN_params[l][k].g ** 2
                _sum += cvBN_params[l][k].b ** 2
                for kX, kY, kZ in ti.ndrange(cvKs[l].shape[1],
                                             cvKs[l].shape[2],
                                             cvKs[l].shape[3]):
                    _sum += cvKs[l][k, kX, kY, kZ] ** 2

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                _sum += dsBs[l][n] ** 2
                for w in range(dsWs[l].shape[1]):
                    _sum += dsWs[l][n, w] ** 2
        
        return tm.sqrt(_sum)

    @ti.kernel
    def _sum_params_grad_l2(self) -> real:
        cvKs = ti.static(self.conv_kernels)
        cvBN_params = ti.static(self.conv_bn_params)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        _sum = 0.
        for l in ti.static(range(len(cvKs))):
            for k in range(cvKs[l].shape[0]):
                _sum += cvBN_params[l].grad[k].g ** 2
                _sum += cvBN_params[l].grad[k].b ** 2
                for kX, kY, kZ in ti.ndrange(cvKs[l].shape[1],
                                             cvKs[l].shape[2],
                                             cvKs[l].shape[3]):
                    _sum += cvKs[l].grad[k, kX, kY, kZ] ** 2

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                _sum += dsBs[l].grad[n] ** 2
                for w in range(dsWs[l].shape[1]):
                    _sum += dsWs[l].grad[n, w] ** 2
        
        return tm.sqrt(_sum)

    @ti.kernel
    def _clear_grads(self):
        cvMs = ti.static(self.conv_maps)
        cvMs_raw = ti.static(self.conv_maps_raw)
        cvKs = ti.static(self.conv_kernels)
        cvBN_params = ti.static(self.conv_bn_params)
        dsOs = ti.static(self.dense_outputs)
        dsOs_raw = ti.static(self.dense_outputs_raw)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        # for n in self._loss:
        self._loss[None] = 0.
        self._loss.grad[None] = 1.

        for O in ti.grouped(cvMs[0]):
            cvMs[0].grad[O] = 0.
        for l in ti.static(range(len(cvMs_raw))):
            for k in range(cvKs[l].shape[0]):
                cvBN_params[l].grad[k].g = 0.
                cvBN_params[l].grad[k].b = 0.
                for kX, kY, kZ in ti.ndrange(cvKs[l].shape[1], 
                                             cvKs[l].shape[2],
                                             cvKs[l].shape[3]):
                    cvKs[l].grad[k, kX, kY, kZ] = 0.
            for O in ti.grouped(cvMs_raw[l]):
                cvMs_raw[l].grad[O] = 0.
            for O in ti.grouped(cvMs[l+1]):
                cvMs[l+1].grad[O] = 0.

        for n in dsOs[0]:
            dsOs[0].grad[n] = 0.
        for l in ti.static(range(len(dsOs_raw))):
            for n in range(dsWs[l].shape[0]):
                dsBs[l].grad[n] = 0.
                dsOs[l+1].grad[n] = 0.
                dsOs_raw[l].grad[n] = 0.
                for w in range(dsWs[l].shape[1]):
                    dsWs[l].grad[n, w] = 0.

    @ti.kernel
    def _forward(self, entry: ndarray):
        cvMs = ti.static(self.conv_maps)
        cvMs_raw = ti.static(self.conv_maps_raw)
        cvKs = ti.static(self.conv_kernels)
        cvBN = ti.static(self._conv_bn)
        cvBN_params = ti.static(self.conv_bn_params)
        dsOs = ti.static(self.dense_outputs)
        dsOs_raw = ti.static(self.dense_outputs_raw)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

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
                for kX, kY, kZ in ti.ndrange(cvKs[l].shape[1],
                                             cvKs[l].shape[2],
                                             cvKs[l].shape[3]):
                    iX = O.x * step - padding + kX  # input X
                    iY = O.y * step - padding + kY  # input Y
                    if (0 <= iX < cvMs[l].shape[0]) and (0 <= iY < cvMs[l].shape[1]):
                        cvMs_raw[l][O] += cvKs[l][O.z, kX, kY, kZ] * cvMs[l][iX, iY, kZ]

            for oZ in range(cvMs_raw[l].shape[2]):
                cvBN[l][oZ].mx = 0.
                cvBN[l][oZ].dx = 0.

            # calculate the empirical mean value of the layer's weighted sum
            for O in ti.grouped(cvMs_raw[l]):
                num_values = cvMs_raw[l].shape[0] * cvMs_raw[l].shape[1] * cvMs_raw[l].shape[2]
                cvBN[l][O.z].mx += cvMs_raw[l][O] / (num_values * self.batch_size)
            # calculate the empirical variance value of the layer's weighted sum
            for O in ti.grouped(cvMs_raw[l]):
                num_values = cvMs_raw[l].shape[0] * cvMs_raw[l].shape[1] * cvMs_raw[l].shape[2]
                cvBN[l][O.z].dx += ((cvMs_raw[l][O] - cvBN[l][O.z].mx) ** 2) / (num_values * self.batch_size)

            # loop over feature map coords (x, y, z)
            # to activate the weighted sum
            for M in ti.grouped(cvMs[l+1]):
                _max = -tm.inf # init max as -inf
                # loop over max pooling cell
                for cX, cY in ti.ndrange(mpX, mpY):
                    rX = M.x * mpX + cX  # raw X
                    rY = M.y * mpY + cY  # raw Y
                    # batch normalization
                    _sum_bn = cvBN_params[l][M.z].norm(cvMs_raw[l][rX, rY, M.z], cvBN[l][M.z].mx, cvBN[l][M.z].dx)
                    # ELU activation + max pooling inlined
                    a = ELU(_sum_bn)
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
                dsOs[l+1][n] = sigmoid(dsOs_raw[l][n] + dsBs[l][n])

    @ti.kernel
    def _compute_loss(self, ideal: ndarray, penalty: real):
        actual = ti.static(self.dense_outputs[-1])

        for n in range(actual.shape[0]):
            self._loss[None] += (BCE(ideal[n], actual[n]) - penalty) / (actual.shape[0] * self.batch_size)

    @ti.kernel
    def _advance(self, lr: real, grad_clip: real):
        cvKs = ti.static(self.conv_kernels)
        cvBN_params = ti.static(self.conv_bn_params)
        dsWs = ti.static(self.dense_weights)
        dsBs = ti.static(self.dense_biases)

        for l in ti.static(range(len(cvKs))):
            for k in range(cvKs[l].shape[0]):
                cvBN_params[l][k].g -= lr * cvBN_params[l].grad[k].g
                cvBN_params[l][k].b -= lr * cvBN_params[l].grad[k].b
                for kX, kY, kZ in ti.ndrange(cvKs[l].shape[1], 
                                             cvKs[l].shape[2],
                                             cvKs[l].shape[3]):
                    dL_dW = cvKs[l].grad[k, kX, kY, kZ] * grad_clip
                    cvKs[l][k, kX, kY, kZ] -= lr * dL_dW

        for l in ti.static(range(len(dsWs))):
            for n in range(dsWs[l].shape[0]):
                dsBs[l][n] -= lr * dsBs[l].grad[n] * grad_clip
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
        self.batch_size = 1
        self._forward(entry)
        
        ti.sync()
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
                        image_array = np.array(image, dtype=np.float32)
                        image_array_aug1 = np.array(image.filter(ImageFilter.BLUR), dtype=np.float32)
                        image_array_aug2 = np.array(image.filter(ImageFilter.EMBOSS), dtype=np.float32)
                        image_array_aug3 = np.array(image.filter(ImageFilter.SMOOTH_MORE), dtype=np.float32)
                        image_array_aug4 = np.array(image.filter(ImageFilter.SHARPEN), dtype=np.float32)

                    imgs.append(image_array)
                    imgs.append(image_array_aug1)
                    imgs.append(image_array_aug2)
                    imgs.append(image_array_aug3)
                    imgs.append(image_array_aug4)
                    lbls.append(image_label)
                    lbls.append(image_label)
                    lbls.append(image_label)
                    lbls.append(image_label)
                    lbls.append(image_label)
                except:
                    pass

        return imgs, lbls
    
    def train(self, dataset_url: str, 
              epochs: int = 1000, 
              batch_size: int = 16,
              learn_rate: float = 0.005, 
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

        history = np.zeros(shape=(epochs // history_interval), dtype=np.float32)
        self.batch_size = batch_size

        imgs, lbls = self._collect_dataset(dataset_url)
        idxs = np.array([np.random.randint(0, len(imgs), size=batch_size) for _ in range(epochs)])

        for epoch in tqdm(range(epochs), 
                          file=sys.stdout, 
                          unit='epochs',
                          desc=f'Training progress: ',
                          colour='green'):
            penalty = self._sum_params_l2() * l2_lambda / self.param_num

            for idx in idxs[epoch]:
                with ti.ad.Tape(loss=self._loss):
                    self._forward(imgs[idx])
                    self._compute_loss(lbls[idx], penalty)

                # params correction
                grad_clip = 1.
                grad_l2 = self._sum_params_grad_l2()
                if grad_l2 >= grad_threshold:
                    grad_clip = grad_threshold / grad_l2

                self._advance(learn_rate, grad_clip)

            history[epoch // history_interval] += self._loss.to_numpy().mean() / history_interval

            if epoch % dump_interval == 0:
                self.dump_params(dump_url)

        return history
    
    def compute_accuracy(self, training_url: str):
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

        groups = os.listdir(training_url)
        groups_num = len(groups)

        results = np.zeros(shape=groups_num, dtype=np.float16)

        for i in tqdm(range(groups_num), 
                      file=sys.stdout, 
                      unit=' classes ',
                      desc='Validation progress: ',
                      colour='blue'):
            group_path = os.path.join(training_url, groups[i])
            group_samples_num = len(os.listdir(group_path))
            group_res = np.zeros(group_samples_num, dtype=np.float16)

            for sample in range(group_samples_num):
                try:
                    with Image.open(os.path.join(group_path, os.listdir(group_path)[sample])) as image:
                        image = image.resize((self.input_layer.size[0], self.input_layer.size[1]))
                        image = image.convert('RGB')
                        image_array = np.array(image, dtype=np.float32)

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
            Net.Layers.Conv(Net.Kernel(256, (3, 3), step=2, padding=1), max_pool=(1, 1)),
        ],
        dense_topology=[
            Net.Layers.Dense(neurons_num=256),
            Net.Layers.Dense(neurons_num=2)
        ]
    )

    print(net.param_num)  # 1093604

    imgs = [Image.open('dataset/training/dew/2208.jpg').resize((64, 64)).convert('RGB'),
            Image.open('dataset/training/dew/2209.jpg').resize((64, 64)).convert('RGB'),
            Image.open('dataset/training/shine/shine1.jpg').resize((64, 64)).convert('RGB'),
            Image.open('dataset/training/shine/shine2.jpg').resize((64, 64)).convert('RGB')]

    for img in imgs:
        arr = np.array(img, dtype=np.float32)
        print(net.predict(arr))

    history = net.train(dataset_url='dataset/training',
                        epochs=1000,
                        batch_size=4,
                        learn_rate=0.005,
                        l2_lambda=0.2,
                        grad_threshold=100.5,
                        history_interval=10,
                        dump_interval=1000000,
                        dump_url='params.net')

    for img in imgs:
        arr = np.array(img, dtype=np.float32)
        print(net.predict(arr))

    plt.plot(history)
    plt.show()
