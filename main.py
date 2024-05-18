import os
import sys
from matplotlib import pyplot as plt
import taichi as ti
import numpy as np
import pickle
import time

from PIL import Image
from tqdm import tqdm

real = ti.float32
number = ti.int32
scalar = lambda: ti.field(dtype=real)
ndarray = ti.types.ndarray(dtype=real, needs_grad=False, boundary='unsafe')
template = ti.template()

ti.init(arch=ti.cuda,
        default_ip=number,
        default_fp=real,
        flatten_if=True,
        random_seed=int(time.time()),
        device_memory_GB=3.5,
        default_gpu_block_dim=512)


@ti.func
def sigmoid(x: real) -> real:
    return 1 / (1 + ti.exp(-x))


@ti.func
def xavier(n_in: number, n_out: number) -> real:
    u = ti.sqrt(6 / (n_in + n_out))
    return 2 * ti.random(real) * u - u


@ti.data_oriented
class Net:
    def __init__(self, input_size, cv_topology, fc_topology) -> None:
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

        for l in ti.static(range(len(cvMs_raw))):
            cfX = ti.static(cvMs_raw[l].shape[0] // cvMs[l+1].shape[0])
            cfY = ti.static(cvMs_raw[l].shape[1] // cvMs[l+1].shape[1])

            for O in ti.grouped(cvMs_raw[l]):
                cvMs_raw[l][O] = 0.
                for fX, fY, fZ in ti.ndrange(cvFs[l].shape[1],
                                             cvFs[l].shape[2],
                                             cvFs[l].shape[3]):
                    iX = O.x + fX
                    iY = O.y + fY
                    cvMs_raw[l][O] += cvFs[l][O.z, fX, fY, fZ] * cvMs[l][iX, iY, fZ] / 255
            
            for O in ti.grouped(cvMs[l+1]):
                _max = 0.
                for mX, mY in ti.ndrange(cfX, cfY):
                    iX = O.x * cfX + mX
                    iY = O.y * cfY + mY
                    _max = max(0, _max, cvMs_raw[l][iX, iY, O.z] + cvBs[l][O.z])
                cvMs[l+1][O] = _max

        for I in ti.grouped(cvMs[-1]):
            idx = I.x * cvMs[-1].shape[1] * cvMs[-1].shape[2] + I.y * cvMs[-1].shape[2] + I.z
            fcOs[0][idx] = cvMs[-1][I]

        for l in ti.static(range(len(fcWs))):
            for n in range(fcWs[l].shape[0]):
                fcOs_raw[l][n] = 0.
                for w in range(fcWs[l].shape[1]):
                    fcOs_raw[l][n] += fcWs[l][n, w] * fcOs[l][w]
            for n in range(fcWs[l].shape[0]):
                fcOs[l+1][n] = sigmoid(fcOs_raw[l][n] + fcBs[l][n])

    @ti.kernel
    def _copy_input(self, entry: ndarray):
        input_map = ti.static(self.cv_maps[0])

        for I in ti.grouped(entry):
            input_map[I] = entry[I]

    @ti.kernel
    def _copy_target(self, ideal: ndarray):
        for n in ideal:
            self._target[n] = ideal[n]

    @ti.kernel
    def _compute_loss(self, epoch: number):
        ideal = ti.static(self._target)
        actual = ti.static(self.fc_outputs[-1])

        for n in actual:
            l = (ideal[n] - actual[n]) ** 2 / actual.shape[0]
            self._loss[None] += l
            self._history[epoch] += l

    @ti.kernel
    def _advance(self):
        cvFs = ti.static(self.cv_filters)
        cvBs = ti.static(self.cv_biases)
        fcWs = ti.static(self.fc_weights)
        fcBs = ti.static(self.fc_biases)
        lr = ti.static(self._learn_rate)

        for l in ti.static(range(len(cvFs))):
            for fl in range(cvFs[l].shape[0]):
                cvBs[l][fl] -= lr * cvBs[l].grad[fl]
                for wX, wY, wZ in ti.ndrange(cvFs[l].shape[1], 
                                             cvFs[l].shape[2],
                                             cvFs[l].shape[3]):
                    cvFs[l][fl, wX, wY, wZ] -= lr * cvFs[l].grad[fl, wX, wY, wZ]

        for l in ti.static(range(len(fcWs))):
            for n in range(fcWs[l].shape[0]):
                fcBs[l][n] -= lr * fcBs[l].grad[n]
                for w in range(fcWs[l].shape[1]):
                    fcWs[l][n, w] -= lr * fcWs[l].grad[n, w]

    def predict(self, entry):
        self.cv_maps[0].from_numpy(entry)
        self._forward()
        
        ti.sync()
        return self.fc_outputs[-1].to_numpy()
    
    def train(self, samples, targets, epochs=1000, batch_size=4, learn_rate=0.1):
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
        # for epoch in range(epochs):
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

            # ti.sync()

        return self._history.to_numpy()

if __name__ == "__main__":
    net = Net(input_size=(64, 64, 3),
              cv_topology=[((30, 30, 24), (3, 3)),
                           ((14, 14, 12), (3, 3))],
              fc_topology=[64, 2])
    print(net.param_count)

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

    history = net.train(samples=dataset, 
                        targets=trues,
                        epochs=1000,
                        batch_size=32,
                        learn_rate=0.01)
    
    # net.dump_params()

    # print(net.predict(inarr1))
    # print(net.predict(inarr2))
    # print(net.predict(inarr3))
    # print(net.predict(inarr4))
    print(net.predict(scaled_images1[0] / 255))
    print(net.predict(scaled_images2[0] / 255))
    print(net.predict(scaled_images1[-1] / 255))
    print(net.predict(scaled_images2[-1] / 255))

    plt.plot(net._history)
    plt.show()
