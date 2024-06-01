import sys
import numpy as np

from PIL import Image
from PySide2.QtGui import QGuiApplication
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtCore import Slot, QObject

from net import Net

class NetBackend(QObject):
    def __init__(self, model_url: str) -> None:
        super().__init__()
        self.model = Net.load(model_url)

    @Slot(str, result=list)
    def predict(self, url: str):
        img = Image.open(url[8:].replace('/', '\\')).convert('RGB')
        _is = self.model.input_layer.size[:2]
        image_array = np.array(img.resize(_is), dtype=np.float32)
        preds = self.model.predict(image_array / 255)
        return preds.tolist()


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)

    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.load('gui/view.qml')

    net = NetBackend("net.model")
    context = engine.rootContext()
    context.setContextProperty("net", net)

    sys.exit(app.exec_())
