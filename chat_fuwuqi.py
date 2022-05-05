import tkinter as tk
from tkinter import scrolledtext
import socket
import threading
from datetime import datetime


def tcp_recv(sock):
    while True:
        str = sock.recv(1024).decode("utf-8")
        show_info(str)


def send_func(sock):
    str = send_msg.get("0.0", "end")
    sock.send(str.encode("utf-8"))
    show_info(str)


def show_info(str):
    now = datetime.now()
    s_time = now.strftime("%Y-%m-%d %H:%M:%S")
    str = str.rstrip()
    if len(str) == 0:
        return -1
    send_msg.delete("0.0", "end")
    temp = s_time + "\n    " + str + "\n"
    show_msg.insert(tk.INSERT, "%s" % temp)


msFont = '微软雅黑'  # 字体
fontSize = 18  # 字体大小
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("127.0.0.1", 8888))
sock.listen(5)

mainWindow = tk.Tk()
mainWindow.title("服务器")
mainWindow.minsize(1000, 400)
show_msg = scrolledtext.ScrolledText(mainWindow, font=(msFont, fontSize))
show_msg.place(width=400, height=250, x=0, y=0)
send_msg = scrolledtext.ScrolledText(mainWindow, font=(msFont, fontSize))
send_msg.place(width=400, height=140, x=0, y=260)
button_send = tk.Button(mainWindow, font=(msFont, fontSize), text="发  送", bg="orange", fg="white",
                        command=lambda: send_func(s))
button_send.place(width=100, height=40, x=300, y=360)

s, addr = sock.accept()
t = threading.Thread(target=tcp_recv, args=(s,))
t.start()
tk.mainloop()