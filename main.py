import os
import sys
import time
import pickle
import numpy as np
import taichi as ti

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
        default_ip=number,
        random_seed=int(time.time()))


@ti.func
def sigmoid(x: real) -> real:
    return 1 / (1 + ti.exp(-x))


@ti.func
def xavier(n_in: number, n_out: number) -> real:
    u = ti.sqrt(6 / (n_in + n_out))

    # init range: [-u, +u]
    return 2 * ti.random(real) * u - u


@ti.data_oriented
class ConvNet:
    def __init__(self, input_size: tuple, 
                 cv_topology: list, 
                 fc_topology: list) -> None:
        """Initialize a CNN instance.

        Args:
            input_size: `tuple` - 3-dimensional size of the input layer.
            cv_topology: `list` - Topology of the convolutional layers. 
                                    Each layer must be a list of a 3-dimensional 
                                    feature map size and a 2-dimensional filter size.
            fc_topology: `list` - Topology of the fully connected layers.
                                    Each layer must be a number of neurons it contains.

        Example::

            >>> net = ConvNet(
            >>>     input_size=(64, 64, 3),
            >>>     cv_topology=[[(30, 30, 6), (3, 3)],
            >>>                  [(14, 14, 3), (3, 3)]],
            >>>     fc_topology=[64, 2]
            >>> )
        """

        self.input_size = input_size
        self.cv_topology = cv_topology
        self.fc_topology = fc_topology

        self.cv_filters = []
        self.cv_biases = []
        self.cv_maps_raw = []
        self.cv_maps = []

        self.fc_outputs = []
        self.fc_outputs_raw = []
        self.fc_weights = []
        self.fc_biases = []

        self._loss = scalar()
        self._target = scalar()

        self._allocate()
        self._init_params()

    def _allocate(self):
        assert self.input_size != None
        assert self.cv_topology != None
        assert self.fc_topology != None

        cv_map = scalar()
        ti.root.dense(ti.ijk, self.input_size).place(cv_map)
        self.cv_maps.append(cv_map)

        for i in range(len(self.cv_topology)):
            filters_num = self.cv_topology[i][0][2]

            map_raw_size = (self.cv_maps[i].shape[0] - self.cv_topology[i][1][0] + 1,
                            self.cv_maps[i].shape[1] - self.cv_topology[i][1][1] + 1)
            map_size = (self.cv_topology[i][0][0], self.cv_topology[i][0][1])

            cv_map_raw = scalar()
            cv_map = scalar()
            cv_biases = scalar()
            cv_filters= scalar()

            map_block = ti.root.dense(ti.k, filters_num)
            map_block.dense(ti.ij, map_raw_size).place(cv_map_raw)
            map_block.dense(ti.ij, map_size).place(cv_map)

            param_block = ti.root.dense(ti.i, filters_num)
            param_block.place(cv_biases)

            filter_block = param_block.dense(ti.jk, self.cv_topology[i][1])
            filter_block.dense(ti.l, self.cv_maps[i].shape[2]).place(cv_filters)

            self.cv_maps_raw.append(cv_map_raw)
            self.cv_maps.append(cv_map)
            self.cv_biases.append(cv_biases)
            self.cv_filters.append(cv_filters)

        fc_output = scalar()
        cv_map_flatten_size = self.cv_topology[-1][0][0] * self.cv_topology[-1][0][1] * self.cv_topology[-1][0][2]
        ti.root.dense(ti.i, cv_map_flatten_size).place(fc_output)
        self.fc_outputs.append(fc_output)

        for i in range(len(self.fc_topology)):
            neurons_num = self.fc_topology[i]
            weights_num = self.fc_outputs[-1].shape[0]

            fc_output = scalar()
            fc_output_raw = scalar()
            fc_biases = scalar()
            fc_weights = scalar()

            neuron_block = ti.root.dense(ti.i, neurons_num) 
            neuron_block.place(fc_output_raw, fc_output, fc_biases)
            neuron_block.dense(ti.j, weights_num).place(fc_weights)

            self.fc_outputs_raw.append(fc_output_raw)
            self.fc_outputs.append(fc_output)
            self.fc_biases.append(fc_biases)
            self.fc_weights.append(fc_weights)

        ti.root.dense(ti.i, self.fc_topology[-1]).place(self._target)
        ti.root.place(self._loss)
        ti.root.lazy_grad()

    @property
    def param_count(self):
        """Return a total number of a CNN instance's parameters
        
        Example::

            >>> net = ConvNet(
            >>>     input_size=(64, 64, 3),
            >>>     cv_topology=[[(30, 30, 6), (3, 3)],
            >>>                  [(14, 14, 3), (3, 3)]],
            >>>     fc_topology=[64, 2]
            >>> )
            >>> print(net.param_count)  # 38159
            """

        total = 0

        for fc_weights in self.fc_weights:
            total += fc_weights.shape[0] * fc_weights.shape[1]

        for fc_biases in self.fc_biases:
            total += fc_biases.shape[0]

        for cv_filters in self.cv_filters:
            total += cv_filters.shape[0] * cv_filters.shape[1] * cv_filters.shape[2] * cv_filters.shape[3]

        for cv_biases in self.cv_biases:
            total += cv_biases.shape[0]

        return total
    
    def dump_params(self, url: str = 'params.net'):
        """Save a CNN instance's parameters.

        Args:
            url: `str` - path to the file to save the parameters.

        Example::

            >>> net.dump_params('params.net')
        """

        cv_biases = []
        cv_filters = []
        fc_biases = []
        fc_weights = []

        for l in range(len(self.fc_topology)):
            fc_biases.append(self.fc_biases[l].to_numpy())
            fc_weights.append(self.fc_weights[l].to_numpy())

        for l in range(len(self.cv_topology)):
            cv_biases.append(self.cv_biases[l].to_numpy())
            cv_filters.append(self.cv_filters[l].to_numpy())

        params = {
            'fc_weights': fc_weights,
            'fc_biases': fc_biases,
            'cv_filters': cv_filters,
            'cv_biases': cv_biases
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

        fc_biases = params['fc_biases']
        fc_weights = params['fc_weights']
        cv_biases = params['cv_biases']
        cv_filters = params['cv_filters']

        for l in range(len(self.fc_topology)):
            self.fc_biases[l].from_numpy(fc_biases[l])
            self.fc_weights[l].from_numpy(fc_weights[l])

        for l in range(len(self.cv_topology)):
            self.cv_biases[l].from_numpy(cv_biases[l])
            self.cv_filters[l].from_numpy(cv_filters[l])

    @ti.kernel
    def _init_params(self):
        cvMs = ti.static(self.cv_maps)
        cvFs = ti.static(self.cv_filters)
        fcWs = ti.static(self.fc_weights)

        # Xavier Initialization -->
        # 
        for l in ti.static(range(len(cvFs))):
            for fl in range(cvFs[l].shape[0]):
                for wX, wY, wZ in ti.ndrange(cvFs[l].shape[1], 
                                             cvFs[l].shape[2],
                                             cvFs[l].shape[3]):
                    cvFs[l][fl, wX, wY, wZ] = xavier(cvMs[l].shape[2], cvFs[l].shape[3])

        for l in ti.static(range(len(fcWs))):
            for n in range(fcWs[l].shape[0]):
                for w in range(fcWs[l].shape[1]):
                    fcWs[l][n, w] = xavier(fcWs[l].shape[1], fcWs[l].shape[0])

    @ti.kernel
    def _clear_grads(self):
        cvMs = ti.static(self.cv_maps)
        cvMs_raw = ti.static(self.cv_maps_raw)
        cvFs = ti.static(self.cv_filters)
        cvBs = ti.static(self.cv_biases)
        fcOs = ti.static(self.fc_outputs)
        fcOs_raw = ti.static(self.fc_outputs_raw)
        fcWs = ti.static(self.fc_weights)
        fcBs = ti.static(self.fc_biases)

        self._loss[None] = 0
        self._loss.grad[None] = 1

        for O in ti.grouped(cvMs[0]):
            cvMs[0].grad[O] = 0.
        for l in ti.static(range(len(cvMs_raw))):
            for fl in range(cvFs[l].shape[0]):
                cvBs[l].grad[fl] = 0.
                for wX, wY, wZ in ti.ndrange(cvFs[l].shape[1], 
                                             cvFs[l].shape[2],
                                             cvFs[l].shape[3]):
                    cvFs[l].grad[fl, wX, wY, wZ] = 0.
            for O in ti.grouped(cvMs_raw[l]):
                cvMs_raw[l].grad[O] = 0.
            for O in ti.grouped(cvMs[l+1]):
                cvMs[l+1].grad[O] = 0.

        for n in fcOs[0]:
            fcOs[0].grad[n] = 0.
        for l in ti.static(range(len(fcOs_raw))):
            for n in range(fcWs[l].shape[0]):
                fcBs[l].grad[n] = 0.
                fcOs[l+1].grad[n] = 0.
                fcOs_raw[l].grad[n] = 0.
                for w in range(fcWs[l].shape[1]):
                    fcWs[l].grad[n, w] = 0.

    @ti.kernel
    def _forward(self):
        cvMs = ti.static(self.cv_maps)
        cvMs_raw = ti.static(self.cv_maps_raw)
        cvFs = ti.static(self.cv_filters)
        cvBs = ti.static(self.cv_biases)
        fcOs = ti.static(self.fc_outputs)
        fcOs_raw = ti.static(self.fc_outputs_raw)
        fcWs = ti.static(self.fc_weights)
        fcBs = ti.static(self.fc_biases)

        # loop over conv layers
        for l in ti.static(range(len(cvMs_raw))):
            # loop over feature raw map coords (x, y, z)
            # to calculate the layer's weighted sum
            for O in ti.grouped(cvMs_raw[l]):
                cvMs_raw[l][O] = 0.
                # loop over filter coords (x, y, z)
                for fX, fY, fZ in ti.ndrange(cvFs[l].shape[1],
                                             cvFs[l].shape[2],
                                             cvFs[l].shape[3]):
                    iX = O.x + fX  # input X
                    iY = O.y + fY  # input Y
                    cvMs_raw[l][O] += cvFs[l][O.z, fX, fY, fZ] * cvMs[l][iX, iY, fZ]
            
            # X compression factor
            cfX = ti.static(cvMs_raw[l].shape[0] // cvMs[l+1].shape[0])
            # Y compression factor
            cfY = ti.static(cvMs_raw[l].shape[1] // cvMs[l+1].shape[1])

            # loop over feature map coords (x, y, z)
            # to activate the weighted sum
            for O in ti.grouped(cvMs[l+1]):
                _max = 0.  # init max as zero
                for mX, mY in ti.ndrange(cfX, cfY):
                    rX = O.x * cfX + mX  # raw X
                    rY = O.y * cfY + mY  # raw Y
                    # ReLU + maxpooling inlined
                    _max = max(_max, cvMs_raw[l][rX, rY, O.z] + cvBs[l][O.z])
                cvMs[l+1][O] = _max

        # flatten the convolutional output
        for I in ti.grouped(cvMs[-1]):
            # flatten index is x * width * depth + y * depth + z
            fI = I.x * cvMs[-1].shape[1] * cvMs[-1].shape[2] + I.y * cvMs[-1].shape[2] + I.z
            fcOs[0][fI] = cvMs[-1][I]

        # loop over fully connected layers
        for l in ti.static(range(len(fcWs))):
            # loop over neurons
            # to calculate the layer's weighted sum
            for n in range(fcWs[l].shape[0]):
                fcOs_raw[l][n] = 0.
                # loop over neuron weights
                for w in range(fcWs[l].shape[1]):
                    fcOs_raw[l][n] += fcWs[l][n, w] * fcOs[l][w]
            # loop over neurons
            # to activate the weighted sum
            for n in range(fcWs[l].shape[0]):
                fcOs[l+1][n] = sigmoid(fcOs_raw[l][n] + fcBs[l][n])

    @ti.kernel
    def _copy_input(self, entry: ndarray):
        for I in ti.grouped(entry):
            self.cv_maps[0][I] = entry[I]

    @ti.kernel
    def _copy_target(self, ideal: ndarray):
        for n in ideal:
            self._target[n] = ideal[n]

    @ti.kernel
    def _compute_loss(self, epoch: number):
        ideal = ti.static(self._target)
        actual = ti.static(self.fc_outputs[-1])

        for n in actual:
            L = (ideal[n] - actual[n]) ** 2 / actual.shape[0]
            self._loss[None] += L
            self._history[epoch] += L

    @ti.kernel
    def _advance(self):
        cvFs = ti.static(self.cv_filters)
        cvBs = ti.static(self.cv_biases)
        fcWs = ti.static(self.fc_weights)
        fcBs = ti.static(self.fc_biases)
        lr = ti.static(self._learn_rate)

        for l in ti.static(range(len(cvFs))):
            for fl in range(cvFs[l].shape[0]):
                # new_bias = old_bias - lr * dL_dB
                cvBs[l][fl] -= lr * cvBs[l].grad[fl]
                for wX, wY, wZ in ti.ndrange(cvFs[l].shape[1], 
                                             cvFs[l].shape[2],
                                             cvFs[l].shape[3]):
                    # new_weight = old_weight - lr * dL_dW
                    cvFs[l][fl, wX, wY, wZ] -= lr * cvFs[l].grad[fl, wX, wY, wZ]

        for l in ti.static(range(len(fcWs))):
            for n in range(fcWs[l].shape[0]):
                # new_bias = old_bias - lr * dL_dB
                fcBs[l][n] -= lr * fcBs[l].grad[n]
                for w in range(fcWs[l].shape[1]):
                    # new_weight = old_weight - lr * dL_dW
                    fcWs[l][n, w] -= lr * fcWs[l].grad[n, w]

    def predict(self, entry: ndarray) -> np.ndarray:
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
        """

        self._copy_input(entry)
        self._forward()
        
        ti.sync()
        return self.fc_outputs[-1].to_numpy()
    
    def train(self, samples, targets, epochs=1000, batch_size=4, learn_rate=0.1):
        """Train a CNN instance.

        Args:
            samples: `ndarray` - samples to train the model.
            targets: `ndarray` - sample markers to train the model.
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

        self._learn_rate = learn_rate
        self._history = scalar()
        ti.root.dense(ti.i, epochs).place(self._history)

        batches = np.random.randint(0, samples.shape[0], size=(epochs * batch_size))
        batches = batches.reshape(epochs, batch_size)

        for epoch in tqdm(range(epochs), 
                          file=sys.stdout, 
                          unit=' epochs ',
                          desc='Training process: ',
                          colour='green'):
            self._clear_grads()
            for sample in batches[epoch]:
                self._copy_input(samples[sample])
                self._copy_target(targets[sample])
                # forward pass
                self._forward()
                self._compute_loss(epoch)
                # backward pass
                self._compute_loss.grad(epoch)
                self._forward.grad()
            # params correction
            self._advance()

            ti.sync()

        return self._history.to_numpy()

if __name__ == "__main__":
    net = ConvNet(
        input_size=(64, 64, 3),
        cv_topology=[[(30, 30, 6), (3, 3)],
                     [(14, 14, 3), (3, 3)]],
        fc_topology=[64, 2]
    )
    print(net.param_count)  # 38159

    # net.load_params()

    dataset_path = 'dataset/cloudy'
    scaled_images1 = []

    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = Image.open(os.path.join(dataset_path, filename))
            image = image.resize((64, 64)).convert('RGB')
            image_array = np.array(image, dtype=np.float32)
            scaled_images1.append(image_array)

    dataset_path = 'dataset/dew'
    scaled_images2 = []

    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = Image.open(os.path.join(dataset_path, filename))
            image = image.resize((64, 64)).convert('RGB')
            image_array = np.array(image, dtype=np.float32)
            scaled_images2.append(image_array)

    dataset = np.array(scaled_images1 + scaled_images2)
    tr1 = [[1., 0.] for _ in range(len(scaled_images1))]
    tr2 = [[0., 1.] for _ in range(len(scaled_images2))]
    trues = np.array(tr1 + tr2, dtype=np.float32)

    print(net.predict(scaled_images1[0]))
    print(net.predict(scaled_images2[0]))
    print(net.predict(scaled_images1[-1]))
    print(net.predict(scaled_images2[-1]))

    history = net.train(samples=dataset[:10], 
                        targets=trues[:10],
                        epochs=100,
                        batch_size=1,
                        learn_rate=0.5)
    
    net.dump_params()

    print(net.predict(scaled_images1[0]))
    print(net.predict(scaled_images2[0]))
    print(net.predict(scaled_images1[-1]))
    print(net.predict(scaled_images2[-1]))

    plt.plot(net._history)
    plt.show()
