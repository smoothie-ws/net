import QtQuick 2.15
import QtQuick.Dialogs 1.0
import QtQuick.Layouts 1.12
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 700
    height: 500
    color: Qt.rgba(0.15, 0.15, 0.15, 1)
    title: "Распознавание погоды на изображениях"

    QtObject {
        id: internal
        property var classes: []

        function updatePreds(preds) {
            for (let i = 0; i < classes.length; i++) {
                classes[i].value = preds[i];
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        anchors.margins: 25
        color: Qt.rgba(0.2, 0.2, 0.2, 1)
        radius: 15
    
        RowLayout {
            id: mainLayout
            anchors.fill: parent
            anchors.margins: 25

            Item {
                id: previewArea
                Layout.fillWidth: true
                Layout.fillHeight: true

                Image {
                    id: previewImage
                    anchors.centerIn: parent
                    width: parent.width - 50
                    height: parent.height - 50
                    fillMode: Image.PreserveAspectFit

                    Behavior on scale {
                        NumberAnimation { 
                            duration: 100
                            easing.type: Easing.OutQuart
                        }
                    }

                    source: "assets/img_icon.png"

                    MouseArea {
                        anchors.fill: parent
                        acceptedButtons: Qt.LeftButton
                        hoverEnabled: true

                        onEntered: parent.scale = 1.1
                        onExited: parent.scale = 1.0
                        onPressed: parent.scale = 0.7

                        onClicked: fileDialog.visible = true
                    }
                }
            }

            ColumnLayout {
                id: predsArea
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 15

                Repeater {
                    model: [
                        "Облачно", 
                        "Роса", 
                        "Туман", 
                        "Град", 
                        "Гроза", 
                        "Дождь", 
                        "Изморозь", 
                        "Песчаная буря", 
                        "Снежно", 
                        "Солнечно"
                    ]

                    delegate: Text {
                        Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
                        Layout.fillHeight: true
                        font.family: "Arial"
                        font.pixelSize: 12
                        text: modelData

                        color: Qt.hsla(0., 0., 1., value)
                        scale: .75 + value * .5

                        property real value: 0.5
                    }

                    onItemAdded: internal.classes.push(item)
                }
            }
        }
    }

    FileDialog {
        id: fileDialog
        title: "Please choose a file"
        folder: shortcuts.home

        onAccepted: {
            previewImage.source = fileUrl;
            internal.updatePreds(net.predict(fileUrl));
        }
    }
}
