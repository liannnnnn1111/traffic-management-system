import os
import cv2
import time
import threading
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from tkinter import *
from tkinter import messagebox, ttk
import subprocess





if __name__ == "__main__":
    try:
        __import__("_libcore")
    except Exception:
        pass



save_dir = "logs"
os.makedirs(save_dir, exist_ok=True)
speed_snapshots_dir = os.path.join(save_dir, "overspeed")
os.makedirs(speed_snapshots_dir, exist_ok=True)

model = YOLO("yolov8m.pt")

running = False
count_data = []
cap = None
video_writer = None
video_filename = ""

# Detection configs
target_classes = ['car', 'motorcycle', 'bicycle', 'truck']
total_counts = {cls: 0 for cls in target_classes}
total_counts['ebike'] = 0

last_seen = {}
vehicle_memory = {}

DETECTION_COOLDOWN = 3
IOU_THRESHOLD = 0.3
speed_limit_kmph = 60
ppm = 10  # pixels per meter

# --- GUI ---
root = Tk()
root.title("Lian AI Traffic Monitor")
root.geometry("460x680")
root.columnconfigure(0, weight=1)

main_frame = Frame(root)
main_frame.grid(sticky="nsew", padx=10, pady=10)
main_frame.columnconfigure(0, weight=1)

# Camera selection
camera_option = StringVar(value="Webcam")
cctv_ip = StringVar(value="192.168.100.125")
cctv_port = StringVar(value="4050")
cctv_user = StringVar()
cctv_pass = StringVar()

Label(main_frame, text="Camera Source:", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
OptionMenu(main_frame, camera_option, "Webcam", "CCTV").grid(row=1, column=0, sticky="ew")

ip_label = Label(main_frame, text="CCTV IP:")
ip_entry = Entry(main_frame, textvariable=cctv_ip, width=50)
port_label = Label(main_frame, text="Port:")
port_entry = Entry(main_frame, textvariable=cctv_port, width=50)
user_label = Label(main_frame, text="Username:")
user_entry = Entry(main_frame, textvariable=cctv_user, width=50)
pass_label = Label(main_frame, text="Password:")
pass_entry = Entry(main_frame, textvariable=cctv_pass, show="*", width=50)

def toggle_cctv_fields(*args):
    is_cctv = camera_option.get() == "CCTV"
    for widget in [ip_label, ip_entry, port_label, port_entry, user_label, user_entry, pass_label, pass_entry]:
        widget.grid() if is_cctv else widget.grid_remove()

camera_option.trace_add("write", toggle_cctv_fields)

ip_label.grid(row=2, column=0, sticky="w")
ip_entry.grid(row=3, column=0, sticky="ew", pady=1)
port_label.grid(row=4, column=0, sticky="w")
port_entry.grid(row=5, column=0, sticky="ew", pady=1)
user_label.grid(row=6, column=0, sticky="w")
user_entry.grid(row=7, column=0, sticky="ew", pady=1)
pass_label.grid(row=8, column=0, sticky="w")
pass_entry.grid(row=9, column=0, sticky="ew", pady=1)
toggle_cctv_fields()

Label(main_frame, text="🚦 Lian AI Monitor", font=("Arial", 16, "bold")).grid(row=10, column=0, pady=10)
Button(main_frame, text="▶ Start Detection", command=lambda: start_detection(), width=30).grid(row=11, column=0, pady=3)
Button(main_frame, text="⏹ Stop Detection (Auto-Save)", command=lambda: stop_detection(), width=30).grid(row=12, column=0, pady=3)
Button(main_frame, text="💾 Save Snapshot", command=lambda: save_snapshot(), width=30).grid(row=13, column=0, pady=3)
Button(main_frame, text="🔄 Reset Count", command=lambda: reset_counts(), width=30).grid(row=14, column=0, pady=3)

Label(main_frame, text="Live Count Table", font=("Arial", 12, "bold")).grid(row=15, column=0, pady=(10, 5))
table_frame = Frame(main_frame)
table_frame.grid(row=16, column=0, sticky="nsew")

table_scroll = Scrollbar(table_frame)
table_scroll.pack(side=RIGHT, fill=Y)

table = ttk.Treeview(table_frame, columns=("Class", "Count"), show="headings", yscrollcommand=table_scroll.set)
table.heading("Class", text="Class")
table.heading("Count", text="Count")
table.column("Class", anchor="center", width=150)
table.column("Count", anchor="center", width=100)
table.pack(fill=BOTH, expand=True)
table_scroll.config(command=table.yview)

# --- Detection Functions ---
def compute_iou(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area else 0.0

def estimate_speed(p1, p2, t1, t2):
    px_distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
    meters = px_distance / ppm
    time_elapsed = t2 - t1
    return (meters / time_elapsed) * 3.6 if time_elapsed else 0

def start_detection():
    global running, cap, video_writer, video_filename
    if running:
        return
    running = True

    # Webcam or CCTV
    if camera_option.get() == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        ip = cctv_ip.get().strip()
        port = cctv_port.get().strip()
        username = cctv_user.get().strip()
        password = cctv_pass.get().strip()

        if not ip or not port:
            messagebox.showerror("Error", "CCTV IP and Port are required.")
            running = False
            return

        url = f"http://{username}:{password}@{ip}:{port}/video" if username and password else f"http://{ip}:{port}/video"
        cap = cv2.VideoCapture(url)

    if not cap or not cap.isOpened():
        messagebox.showerror("Error", "Unable to open camera")
        running = False
        return

    width, height = 640, 360
    now_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    video_filename = os.path.join(save_dir, f"traffic_video_{now_str}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 5.0, (width, height))

    threading.Thread(target=detect_objects).start()

def stop_detection():
    global running, cap, video_writer
    running = False
    if cap:
        cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    if any(total_counts.values()):
        now = datetime.now()
        count_data.append([now.strftime("%Y-%m-%d %H:%M:%S"), total_counts.copy()])

    if count_data:
        date_str = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        filename = os.path.join(save_dir, f"traffic_{date_str}.csv")
        rows = []
        for timestamp, counts in count_data:
            row = {'Timestamp': timestamp}
            row.update(counts)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        messagebox.showinfo("Exported", f"Saved:\n📁 {filename}\n🎥 {video_filename}")
        subprocess.Popen(f'explorer "{os.path.abspath(save_dir)}"')
    else:
        messagebox.showinfo("Info", "No data to export.")

def detect_objects():
    global cap, running, video_writer, total_counts, last_seen, vehicle_memory
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))  
        timestamp = time.time()
        results = model(frame)[0]
        boxes = results.boxes
        names = results.names

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            label = names[cls]
            if label not in target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            track_id = f"{label}_{i}"
            speed = 0
            if track_id in vehicle_memory:
                prev_center, prev_time = vehicle_memory[track_id]
                speed = estimate_speed(prev_center, center, prev_time, timestamp)
            vehicle_memory[track_id] = (center, timestamp)

            is_overspeed = speed > speed_limit_kmph
            color = (0, 0, 255) if is_overspeed else (0, 255, 0)
            label_text = f"{label} OVERSPEED {int(speed)} km/h" if is_overspeed else f"{label} {int(speed)} km/h"

            if is_overspeed:
                now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                snapshot_path = os.path.join(speed_snapshots_dir, f"overspeed_{now_str}.jpg")
                cv2.imwrite(snapshot_path, frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            already_counted = False
            if label in last_seen:
                for ts, (cx, cy, cw, ch) in last_seen[label]:
                    iou = compute_iou((x1, y1, w, h), (cx - cw // 2, cy - ch // 2, cw, ch))
                    if timestamp - ts < DETECTION_COOLDOWN and iou > IOU_THRESHOLD:
                        already_counted = True
                        break

            if not already_counted:
                total_counts[label] += 1
                last_seen.setdefault(label, []).append((timestamp, (center[0], center[1], w, h)))

            if label == 'bicycle' and box.conf[0] > 0.4:
                total_counts['ebike'] += 1

        if video_writer:
            video_writer.write(frame)

        update_table()
        cv2.imshow("Lian AI Monitor (press Q to hide)", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    if cap:
        cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def update_table():
    for item in table.get_children():
        table.delete(item)
    for name, count in total_counts.items():
        table.insert('', 'end', values=(name.capitalize(), count))

def save_snapshot():
    if not running:
        messagebox.showwarning("Warning", "Detection is not running.")
        return
    now = datetime.now()
    count_data.append([now.strftime("%Y-%m-%d %H:%M:%S"), total_counts.copy()])
    update_table()

def reset_counts():
    global total_counts
    for cls in total_counts:
        total_counts[cls] = 0
    update_table()

# --- Run GUI ---
root.mainloop()
