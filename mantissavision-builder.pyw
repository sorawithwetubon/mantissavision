import os
import sys
import threading
import torch
import tkinter
import torch.nn
from tkinter import ttk
from tkinter import messagebox
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from win11toast import notify

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

window = tkinter.Tk()
window.title("Mantissa Vision - Image Classification Builder")
window.geometry("494x294")
window.resizable(False, False)
def on_close():
    os._exit(0)
window.protocol("WM_DELETE_WINDOW", on_close)

elementlabel_datasetclasses = tkinter.Label(window, text="Dataset Classes", font=("Arial", 10, "bold"))
elementlabel_datasetclasses.place(anchor='nw', x=10, y=10)
elementlistbox_classlist = tkinter.Listbox(window, width=50, height=15)

data_datasetready = True
data_validationready = True
data_validationlist = True

try:
    dataset_class = datasets.ImageFolder(root="dataset").classes
    for i in dataset_class:
        elementlistbox_classlist.insert(tkinter.END, i)
    string_classlist = f"Totals : {len(dataset_class)}"
    elementlabel_classlist = tkinter.Label(window, text=string_classlist, font=("Arial", 10))
    elementlabel_classlist.place(x=320, y=8, anchor="ne")
    if len(dataset_class) == 1:
        elementlistbox_classlist.insert(tkinter.END, " ")
        elementlistbox_classlist.insert(tkinter.END, "Not enough classes in the dataset,")
        elementlistbox_classlist.insert(tkinter.END, "Please add more class to the 'dataset' folder.")
        data_datasetready = False
except:
    elementlistbox_classlist.insert(tkinter.END, "No dataset found,")
    elementlistbox_classlist.insert(tkinter.END, "Please add dataset to the 'dataset' folder.")
    data_datasetready = False
elementlistbox_classlist.place(x=12, y=38, anchor='nw')

try:
    validation_class = datasets.ImageFolder(root="validationset").classes
    validation_classamout = len(validation_class)
    if validation_classamout == 1:
        elementlistbox_classlist.insert(tkinter.END, " ")
        elementlistbox_classlist.insert(tkinter.END, "Not enough classes in the validationset,")
        elementlistbox_classlist.insert(tkinter.END, "Please add more class to the 'validationset' folder.")
        data_validationready = False
    
except:
    elementlistbox_classlist.insert(tkinter.END, " ")
    elementlistbox_classlist.insert(tkinter.END, "Recommend,")
    elementlistbox_classlist.insert(tkinter.END, "Please add validationset to the 'validationset' folder")
    elementlistbox_classlist.insert(tkinter.END, "for better model performance.")
    data_validationready = True
    data_validationlist = False

if data_validationready == True and data_datasetready == True and data_validationlist != False:
    if dataset_class != validation_class:
        elementlistbox_classlist.insert(tkinter.END, " ")
        elementlistbox_classlist.insert(tkinter.END, "Mismatched classes in the dataset and validationset,")
        elementlistbox_classlist.insert(tkinter.END, "Please check the classes in the 'dataset' and 'validationset' folders.")
        data_datasetready = False
        data_validationready = False


elementlabel_systeminfo = tkinter.Label(window, text="Device Info", font=("Arial", 10, "bold"))
elementlabel_systeminfo.place(x=330, y=8, anchor="nw")
if torch.cuda.is_available():
    data_device = "CUDA"
else:
    data_device = "CPU"
elementlabel_device = tkinter.Label(window, text=f"Device : {data_device}", font=("Arial", 10))
elementlabel_device.place(x=330, y=32, anchor="nw")

elementlabel_epochprogress = tkinter.Label(window, text="Progress", font=("Arial", 10, "bold"))
elementlabel_epochprogress.place(x=330, y=60, anchor="nw")
elementprogressbar_epochprogress = ttk.Progressbar(window, orient="horizontal", length=140, mode="determinate", takefocus=True, maximum=100)
elementprogressbar_epochprogress['value'] = 0
elementprogressbar_epochprogress.place(x=334, y=84, anchor="nw")
elementlabel_epochprogresstime = tkinter.Label(window, text="0 / 0", font=("Arial", 10))
elementlabel_epochprogresstime.place(x=478, y=60, anchor="ne")
def onlynumbers(text):
    return text.isdigit() or text == ""
vcmd = (window.register(onlynumbers), "%P")

data_epochtime = tkinter.StringVar()
elementlabel_epochtime = tkinter.Label(window, text="Epoch Times", font=("Arial", 10, "bold"))
elementlabel_epochtime.place(x=330, y=116, anchor="nw")
elemententry_epochtime = tkinter.Entry(window, textvariable=data_epochtime, font=("Arial", 10), width=11, validate="key", validatecommand=vcmd)
elemententry_epochtime.place(x=334, y=140, anchor="nw")

elementlabel_console = tkinter.Label(window, text="Status", font=("Arial", 10, "bold"))
elementlabel_console.place(x=330, y=170, anchor="nw")
elementtext_console = tkinter.Text(window, width=17, height=3)
elementtext_console.insert(tkinter.END, "Idle\n")
elementtext_console.config(state=tkinter.DISABLED)
elementtext_console.place(x=334, y=194, anchor='nw')

def startbuild():
    if data_validationready == True and data_datasetready == True:
        try:
            if data_epochtime.get() == "" or data_epochtime.get() == "0":
                data_epochtime.set("5")
            data_epochtimeget = int(data_epochtime.get())
            data_epochprogress = 0
            elementlabel_epochprogresstime.config(text=f"{data_epochprogress} / {data_epochtimeget}")
            elementprogressbar_epochprogress.config(maximum=data_epochtimeget)
            elemententry_epochtime.config(state=tkinter.DISABLED)
            elementtext_console.config(state=tkinter.NORMAL)
            elementtext_console.delete("1.0", tkinter.END)
            elementtext_console.insert(tkinter.END, f"Building...\n")
            elementtext_console.insert(tkinter.END, f"Processing may be slower with a large dataset.")
            elementtext_console.config(state=tkinter.DISABLED)
            elementbutton_startbuild.config(text="  Building...  ", state="disabled")
            window.update_idletasks()
            
            model_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            validation_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            model_dataset = datasets.ImageFolder(root="dataset",transform=model_transform)
            try:
                datasets.ImageFolder(root="validationset").classes
                validation_dataset = datasets.ImageFolder(root="validationset",transform=validation_transform)
            except Exception as e:
                validation_dataset = datasets.ImageFolder(root="dataset",transform=validation_transform)

            model_loader = DataLoader(model_dataset, batch_size=32, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=32)

            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            classnumber = len(model_dataset.classes)
            model.fc = torch.nn.Linear(model.fc.in_features, classnumber)
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(model_device)
            model_criterion = torch.nn.CrossEntropyLoss()
            model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            def trainepoch(model, loader, optimizer, criterion):
                model.train()
                total_loss = 0
                for images, labels in loader:
                    images, labels = images.to(model_device), labels.to(model_device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                return total_loss / len(loader)

            def evaluate(model, loader):
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in loader:
                        images, labels = images.to(model_device), labels.to(model_device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                return correct / total
            
            bestaccurate = 0.0
            for epoch in range(data_epochtimeget):
                trainloss = trainepoch(model, model_loader, model_optimizer, model_criterion)
                elementtext_console.config(state=tkinter.NORMAL)
                elementtext_console.delete("1.0", tkinter.END)
                elementtext_console.insert(tkinter.END, f"Trainloss {epoch + 1} : \n{trainloss}\n")
                elementtext_console.config(state=tkinter.DISABLED)
                valueaccurate = evaluate(model, validation_loader)
                elementtext_console.config(state=tkinter.NORMAL)
                elementtext_console.delete("1.0", tkinter.END)
                elementtext_console.insert(tkinter.END, f"Accurate {epoch + 1} : \n{valueaccurate}")
                elementtext_console.config(state=tkinter.DISABLED)
                elementlabel_epochprogresstime.config(text=f"{epoch + 1} / {data_epochtimeget}")
                elementprogressbar_epochprogress['value'] = epoch + 1
                window.update_idletasks()
                elementtext_console.config(state=tkinter.DISABLED)
                if valueaccurate > bestaccurate:
                    bestaccurate = valueaccurate
                    torch.save(model.state_dict(), 'mantissavision-model.pth') 
            notify("Mantissa Vision - Image Classification Builder", "Building Complete, Application will restart.")
            script = os.path.abspath(sys.argv[0])
            os.execl(sys.executable, sys.executable, script, *sys.argv[1:])

        except Exception as error:
            messagebox.showinfo(
                "Mantissa Vision - Image Classification Builder",
                f"Error: {error}"
            )
            script = os.path.abspath(sys.argv[0])
            os.execl(sys.executable, sys.executable, script, *sys.argv[1:])
    else:
        messagebox.showinfo(
            "Mantissa Vision - Image Classification Builder",
            f"dataset or validationset not ready, Please check data and start application again."
        )
        os._exit(0)
def startbuild_thread():
    threading.Thread(target=startbuild).start()
os.makedirs('dataset', exist_ok=True)
os.makedirs('validationset', exist_ok=True)
elementbutton_startbuild = tkinter.Button(window, text="  Start Build  ", command=startbuild_thread)
elementbutton_startbuild.place(x=333, y=255, anchor="nw")
elementlabel_version = tkinter.Label(window, text="V.25A0", font=("Arial", 10))
elementlabel_version.place(x=478, y=257, anchor="ne")
window.mainloop()