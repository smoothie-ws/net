import os
import sys
import pickle
import numpy as np
import taichi as ti

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

real = ti.f64
number = ti.i32
template = ti.template()
scalar = lambda: ti.field(dtype=real)
ndarray = ti.types.ndarray(dtype=real)

ti.init(arch=ti.cuda,
        default_fp=real,
        default_ip=number)


@ti.func
def sigmoid(x: real) -> real:
    return 1 / (1 + ti.exp(-x))


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
            fc_weights = scalar()
            fc_biases = scalar()

            neuron_block = ti.root.dense(ti.i, neurons_num) 
            neuron_block.place(fc_output_raw, fc_output, fc_biases)
            neuron_block.dense(ti.j, weights_num).place(fc_weights)

            self.fc_outputs_raw.append(fc_output_raw)
            self.fc_outputs.append(fc_output)
            self.fc_biases.append(fc_biases)
            self.fc_weights.append(fc_weights)

        ti.root.dense(ti.i, self.fc_topology[-1]).place(self._loss)
        ti.root.dense(ti.i, self.fc_topology[-1]).place(self._target)
        ti.root.lazy_grad()

    @property
    def param_count(self):
        """Return a total number of a CNN instance's parameters
        
        Example::

            >>> net = ConvNet(
            >>>     input_size=(64, 64, 3),
            >>>     cv_topology=[[(30, 30, 32), (3, 3)],
            >>>                  [(14, 14, 64), (5, 5)]],
            >>>     fc_topology=[512, 32, 4]
            >>> )
            >>> print(net.param_count)
            >>> 6491748
            """

        return self._param_count()
    
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
        cvFs = ti.static(self.cv_filters)
        fcWs = ti.static(self.fc_weights)

        for l in ti.static(range(len(cvFs))):
            u = ti.sqrt(6 / (cvFs[l].shape[0] + cvFs[l].shape[3]))
            for F in ti.grouped(cvFs[l]):
                cvFs[l][F] = 2 * ti.random(real) * u - u

        for l in ti.static(range(len(fcWs))):
            u = ti.sqrt(6 / (fcWs[l].shape[0] + fcWs[l].shape[1]))
            for W in ti.grouped(fcWs[l]):
                fcWs[l][W] = 2 * ti.random(real) * u - u

    @ti.kernel
    def _param_count(self) -> number:
        cvFs = ti.static(self.cv_filters)
        cvBs = ti.static(self.cv_biases)
        fcWs = ti.static(self.fc_weights)
        fcBs = ti.static(self.fc_biases)

        total = 0

        for l in ti.static(range(len(fcWs))):
            total += fcWs[l].shape[0] * fcWs[l].shape[1]

        for l in ti.static(range(len(fcBs))):
            total += fcBs[l].shape[0]

        for l in ti.static(range(len(cvFs))):
            total += cvFs[l].shape[0] * cvFs[l].shape[1] * cvFs[l].shape[2] * cvFs[l].shape[3]

        for l in ti.static(range(len(cvBs))):
            total += cvBs[l].shape[0]

        return total
    
    @ti.kernel
    def _param_sqsum(self) -> real:
        cvFs = ti.static(self.cv_filters)
        cvBs = ti.static(self.cv_biases)
        fcWs = ti.static(self.fc_weights)
        fcBs = ti.static(self.fc_biases)

        _sum = 0.
        for l in ti.static(range(len(cvFs))):
            for fL in range(cvFs[l].shape[0]):
                _sum += cvBs[l][fL] ** 2
                for fX, fY, fZ in ti.ndrange(cvFs[l].shape[1],
                                             cvFs[l].shape[2],
                                             cvFs[l].shape[3]):
                    _sum += cvFs[l][fL, fX, fY, fZ] ** 2

        for l in ti.static(range(len(fcWs))):
            for n in range(fcWs[l].shape[0]):
                _sum += fcBs[l][n] ** 2
                for w in range(fcWs[l].shape[1]):
                    _sum += fcWs[l][n, w] ** 2

        return _sum
    
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

        for n in self._loss:
            self._loss[n] = 0.
            self._loss.grad[n] = 1.

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
    def _compute_loss(self):
        ideal = ti.static(self._target)
        actual = ti.static(self.fc_outputs[-1])

        for n in range(actual.shape[0]):
            L = ideal[n] * ti.log(actual[n]) + (1 - ideal[n]) * ti.log(1 - actual[n])
            self._loss[n] = -(L + self._penalty) 

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
            >>> [0.17293, 0.97124, ... ]
        """

        self.cv_maps[0].from_numpy(entry / 255)
        self._forward()
        
        ti.sync()
        return self.fc_outputs[-1].to_numpy()
    
    def train(self, training_url: str, 
              epochs: int = 1000, 
              history_interval: int = 10, 
              dump_interval: int = 100, 
              dump_url: str = 'params.net',
              learn_rate: float = 0.1, 
              l2_lambda: float = 0.01):
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

        self._learn_rate = learn_rate
        self._penalty = 0.0

        history = np.zeros(shape=(epochs // history_interval), dtype=np.float32)

        imgs = []
        lbls = []

        groups = os.listdir(training_url)
        groups_num = len(groups)
        param_count = self.param_count

        for i in tqdm(range(groups_num), 
                      file=sys.stdout, 
                      unit=' classes ',
                      desc='Collecting dataset: ',
                      colour='green'):
            group_path = os.path.join(training_url, groups[i])

            for filename in os.listdir(group_path):
                image_label = np.array([0. for _ in range(groups_num)], dtype=np.float64)
                image_label[i] = 1.

                try:
                    with Image.open(os.path.join(group_path, filename)) as image:
                        image_array = np.array(image.resize((64, 64)).convert('RGB'), dtype=np.float64) / 255
                except:
                    pass

                imgs.append(image_array)
                lbls.append(image_label)

        idxs = np.random.randint(0, len(imgs), size=epochs)

        for epoch in tqdm(range(epochs), 
                          file=sys.stdout, 
                          unit=' epochs ',
                          desc=f'Training progress: ',
                          colour='green'):
            self._penalty = (self._param_sqsum() / param_count) * l2_lambda
            self.cv_maps[0].from_numpy(imgs[idxs[epoch]])
            self._target.from_numpy(lbls[idxs[epoch]])

            self._clear_grads()
            # forward pass
            self._forward()
            self._compute_loss()
            # backward pass
            self._compute_loss.grad()
            self._forward.grad()
            # params correction
            self._advance()

            history[epoch // history_interval] += self._loss.to_numpy().mean() / history_interval

            if epoch % dump_interval == 0:
                self.dump_params(dump_url)

        return history
    
    def compute_accuracy(self, training_url: str):
        """Calculate the approximate accuracy of the model.

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

        results = np.zeros(shape=groups_num, dtype=np.float64)

        for i in tqdm(range(groups_num), 
                      file=sys.stdout, 
                      unit=' classes ',
                      desc='Validation progress: ',
                      colour='blue'):
            group_path = os.path.join(training_url, groups[i])
            group_samples_num = len(os.listdir(group_path))
            group_res = np.zeros(group_samples_num, dtype=np.float64)

            for sample in range(group_samples_num):
                try:
                    with Image.open(os.path.join(group_path, os.listdir(group_path)[sample])) as image:
                        image_array = np.array(image.resize((64, 64)).convert('RGB'), dtype=np.float32) / 255
                        self.cv_maps[0].from_numpy(image_array)
                        self._forward()
                        group_res[sample] = self.fc_outputs[-1].to_numpy().argmax()
                except:
                    pass

            results[i] = np.sum(group_res == i) / group_samples_num

        return results.mean()


if __name__ == "__main__":
    net = ConvNet(
        input_size=(64, 64, 3),
        cv_topology=[[(30, 30, 32), (3, 3)],
                     [(14, 14, 64), (5, 5)]],
        fc_topology=[512, 64, 10]
    )
    print(net.param_count)  # 6508682
    
    # net.load_params()

    history = net.train(training_url='dataset/training',
                        epochs=10000,
                        history_interval=100,
                        dump_interval=250,
                        dump_url='params.net',
                        learn_rate=0.005,
                        l2_lambda=0.05)

    print(net.compute_accuracy('dataset/training'))

    plt.plot(history)
    plt.show()
