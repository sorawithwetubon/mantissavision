import os
import sys
import torch
import torch.nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import tkinter
from tkinter import messagebox
from PIL import Image, ImageTk
from flask import Flask, Response
from flask_socketio import SocketIO

import threading 
import cv2
import webbrowser
import socket

window = tkinter.Tk()
window.title("Mantissa Vision - Runner")
window.geometry("976x516")
window.resizable(False, False)
def on_close():
    os._exit(0)
window.protocol("WM_DELETE_WINDOW", on_close)
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

def getlocalip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

elementlabel_predictionresult = tkinter.Label(window, text="Prediction Result : Idle", font=("Arial", 10))
elementlabel_predictionresult.place(x=636, y=8, anchor="ne")

elementlabel_datasetclasses = tkinter.Label(window, text="Camera", font=("Arial", 10, "bold"))
elementlabel_datasetclasses.place(anchor='nw', x=10, y=8)
cameraframe_image = Image.open("resource-cover.png")
cameraframe_photo = ImageTk.PhotoImage(cameraframe_image)
cameraframe = tkinter.Label(window, image=cameraframe_photo, borderwidth=1, relief="solid", width=618 , height=464)
cameraframe.place(x=12, y=38, anchor='nw')

elementlabel_systeminfo = tkinter.Label(window, text="Device Info", font=("Arial", 10, "bold"))
elementlabel_systeminfo.place(x=646, y=8, anchor="nw")
if torch.cuda.is_available():
    data_device = "CUDA"
else:
    data_device = "CPU"
elementlabel_device = tkinter.Label(window, text=f"Device : {data_device}", font=("Arial", 10))

elementlabel_device.place(x=646, y=32, anchor="nw")

elementlabel_cameraselection = tkinter.Label(window, text="Camera Selection", font=("Arial", 10, "bold"))
elementlabel_cameraselection.place(x=647, y=60, anchor="nw")
data_cameraselection = tkinter.StringVar()
def only_numbers(text):
    return text.isdigit() or text == ""
vcmd = (window.register(only_numbers), "%P")
elemententry_cameraselection = tkinter.Entry(window, textvariable=data_cameraselection, font=("Arial", 10), width=16, validate="key", validatecommand=vcmd)
elemententry_cameraselection.insert(0,"0")
elemententry_cameraselection.place(x=651, y=84, anchor="nw")

elementlabel_datasetclasses = tkinter.Label(window, text="Dataset Classes", font=("Arial", 10, "bold"))
elementlabel_datasetclasses.place(x=648, y=116, anchor="nw")
elementlistbox_classlist = tkinter.Listbox(window, width=50, height=11)
try:
    data_class = datasets.ImageFolder(root="dataset").classes
    for i in data_class:
        elementlistbox_classlist.insert(tkinter.END, i)
    string_classlist = f"Totals : {len(data_class)}"
    elementlabel_classlist = tkinter.Label(window, text=string_classlist, font=("Arial", 10))
    elementlabel_classlist.place(x=958, y=116, anchor="ne")
except:
    elementlistbox_classlist.insert(tkinter.END, "No dataset found")
    elementlistbox_classlist.insert(tkinter.END, "Please add a dataset to the 'dataset' folder")
elementlistbox_classlist.place(x=651, y=140, anchor='nw')

data_networkport = tkinter.StringVar()
elementlabel_networkport = tkinter.Label(window, text="Web Remote Port", font=("Arial", 10, "bold"))
elementlabel_networkport.place(x=648, y=334, anchor="nw")
elemententry_networkportselection = tkinter.Entry(window, textvariable=data_networkport, font=("Arial", 10), width=16, validate="key", validatecommand=vcmd)
elemententry_networkportselection.insert(0,"7500")
elemententry_networkportselection.place(x=651, y=358, anchor="nw")

data_networkip = tkinter.StringVar()
elementlabel_networkip = tkinter.Label(window, text="Web Remote", font=("Arial", 10, "bold"))
elementlabel_networkip.place(x=648, y=392, anchor="nw")
elementlabel_networkipnumber = tkinter.Label(window, text="Idle", font=("Arial", 10))
elementlabel_networkipnumber.place(x=648, y=412, anchor="nw")

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <style>
            @import url('https://fonts.cdnfonts.com/css/jetbrains-mono');
            body {
                padding: 20px;
            }

            .text-header {
                font-size: 16px;
                font-weight: 600;
                font-family: 'JetBrains Mono', sans-serif;  
                margin-bottom: 20px;
            }

            .image-frame {
                border-radius: 10px;
                border: solid 1px #000000;
                margin-bottom: 10px;
                margin-top: 5px;
            }

            .text-title {
                font-size: 14px;
                font-weight: 600;
                font-family: 'JetBrains Mono', sans-serif;  
                margin-bottom: 5px;
            }

            .text-content {
                font-size: 14px;
                font-weight: 400;
                font-family: 'JetBrains Mono', sans-serif;  
                margin-bottom: 20px 
            }

            .grid {
                display: grid;
                grid-template-columns: 650px 1fr
            }
        </style>
        <html>
            <head>
                <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
                <script>
                    document.addEventListener("DOMContentLoaded", () => {
                        const socket = io();
                        socket.on("socket_prediction", (data) => {
                            document.getElementById("textprediction").innerText = data.text;
                        });

                        socket.on("socket_deviceinfo", (data) => {
                            document.getElementById("textdeviceinfo").innerText = data.text;
                        });

                        socket.on("socket_classlist", (data) => {
                            const htmlOutput = data.text.join('<br>');
                            document.getElementById("textclasslist").innerHTML = htmlOutput;
                        });
                        
                    });
                </script>
            </head>
        <body>
            <div class="text-header">Mantissa Vision - Runner (Web Remote)</div>
            <div class="grid">
                <div>
                    <div class="text-title">Camera</div>
                    <img class="image-frame" src="/capturefeed" width="618" height="464"/>
                    <div class="text-content" id="textprediction">Fetching...</div>
                </div>
                  <div>
                    <div class="text-title">Device Info</div>
                    <div class="text-content" id="textdeviceinfo">Fetching...</div>
                    <div class="text-title"> Dataset Classes</div>
                    <div class="text-content" id="textclasslist">Fetching...</div>
                </div>
            </div>
        </body>
    </html>
    '''
   
def emitcapturefeed():
    while True:
        success,frame = capture.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            framebyte = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + framebyte + b'\r\n')

@app.route('/capturefeed')
def capturefeed():
    return Response(emitcapturefeed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def runflask(port):
    app.run(host="0.0.0.0", port=port, debug=False)

def startserver(port):
    threading.Thread(target=runflask, args=(port,), daemon=True).start()

def startcamera():
    try:
        if data_cameraselection.get() == "":
            data_cameraselection.set("0")
        data_cameraselectionget = int(data_cameraselection.get())

        if data_networkport.get() == "":
            data_networkport.set("7500")
        data_networkportget = int(data_networkport.get())
        global capture
        capture = cv2.VideoCapture(data_cameraselectionget)
        if not capture.isOpened():
            messagebox.showinfo("Mantissa Vision - Runner", "Error: Could not open camera.")
            os.execl(sys.executable, sys.executable, *sys.argv)
        elemententry_cameraselection.config(state=tkinter.DISABLED)
        elemententry_networkportselection.config(state=tkinter.DISABLED)
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder("dataset", transform=transform)
        classnames = dataset.classes
        numclasses = len(classnames)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, numclasses)
        model.load_state_dict(torch.load("mantissavision-model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 618)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 464)
        def updateframe():
            global predictionresulttext
            predictionresulttext = ""
            ret,frame = capture.read()
            if ret:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                input_tensor = transform(pil_image).unsqueeze(0)
                imgtk = ImageTk.PhotoImage(image=pil_image)
                cameraframe.imgtk = imgtk
                cameraframe.configure(image=imgtk)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0].cpu(), dim=0)
                    conf, pred_idx = torch.max(probabilities, 0)
                    pred_class = classnames[pred_idx]
                predictionresulttext = f"Prediction Result : {pred_class} ({conf.item()*100:.0f}%)"
                elementlabel_predictionresult.config(text=predictionresulttext)
                socketio.emit('socket_prediction', {'text': predictionresulttext})
                socketio.emit('socket_deviceinfo', {'text': f"Device : {data_device}"})
                socketio.emit('socket_classlist', {'text': data_class})

            cameraframe.after(10, updateframe)
        updateframe()
        startserver(data_networkportget)
        button.config(text="  Stop Camera  ")
        elementlabel_networkipnumber.config(text=f"http://{getlocalip()}:{data_networkportget}", fg="blue", cursor="hand2")
        elementlabel_networkipnumber.bind("<Button-1>", lambda e: webbrowser.open_new(f"http://{getlocalip()}:{data_networkportget}"))
        window.update_idletasks()


    except Exception as e:
        messagebox.showinfo("Mantissa Vision - Runner", f"Error: {e}")
        script = os.path.abspath(sys.argv[0])
        os.execl(sys.executable, sys.executable, script, *sys.argv[1:])

def startcamera_thread():
    current_text = button.cget("text")
    if current_text == "  Start Camera  ":
        threading.Thread(target=startcamera).start()
    else:
        script = os.path.abspath(sys.argv[0])
        os.execl(sys.executable, sys.executable, script, *sys.argv[1:])
os.makedirs('dataset', exist_ok=True)
os.makedirs('validationset', exist_ok=True)

button = tkinter.Button(window, text="  Start Camera  ", command=startcamera_thread)
button.place(x=650, y=478, anchor="nw")
elementlabel_version = tkinter.Label(window, text="Version 25A0", font=("Arial", 10))
elementlabel_version.place(x=876, y=480, anchor="nw")

window.mainloop()